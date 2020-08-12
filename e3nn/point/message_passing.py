# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch
import torch_geometric

# dev
import math
from itertools import accumulate
from functools import partial

from typing import List

from e3nn import rs
from e3nn import tensor_message
from e3nn.rs import TY_RS_LOOSE
from e3nn.o3 import get_flat_coupling_coefficients
from e3nn.rsh import spherical_harmonics_xyz

from abc import ABC


class Convolution(torch_geometric.nn.MessagePassing):
    def __init__(self, kernel):
        super(Convolution, self).__init__(aggr='add', flow='target_to_source')
        self.kernel = kernel

    def forward(self, features, edge_index, edge_r, size=None, n_norm=1, groups=1):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_target - position_source
        :param size: (n_source, n_target) or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_source, dim(Rs_out)]
        """
        k = self.kernel(edge_r)
        k.div_(n_norm ** 0.5)
        return self.propagate(edge_index, size=size, x=features, k=k, groups=groups)

    def message(self, x_j, k, groups):
        N = x_j.shape[0]
        cout, cin = k.shape[-2:]
        x_j = x_j.view(N, groups, cin)  # Rs_tp1
        if k.shape[0] == 0:  # https://github.com/pytorch/pytorch/issues/37628
            return torch.zeros(0, groups * cout)
        if k.dim() == 4 and k.shape[1] == groups:  # kernel has group dimension
            return torch.einsum('egij,egj->egi', k, x_j).reshape(N, groups * cout)
        return torch.einsum('eij,egj->egi', k, x_j).reshape(N, groups * cout)


class TensorPassingContext(torch.nn.Module, ABC):
    def __init__(self, representations: List[TY_RS_LOOSE]):
        super().__init__()

        self.user_representations = [rs.convention(Rs) for Rs in representations]
        self.internal_representations = [rs.simplify(rs.sort(Rs)[0]) for Rs in self.user_representations]
        self.input_representations = self.internal_representations[:-1]
        self.output_representations = self.internal_representations[1:]

        max_l_per_layer = [rs.lmax(Rs) for Rs in self.internal_representations]

        self.max_l_out = rs.lmax(max_l_per_layer[:-1])
        self.max_l_in = rs.lmax(max_l_per_layer[1:])
        self.max_l = rs.lmax([l_in + l_out for (l_in, l_out) in zip(max_l_per_layer[:-1], max_l_per_layer[1:])])

        coupling_coefficients, coupling_coefficients_offsets = get_flat_coupling_coefficients(self.max_l_out, self.max_l_in)
        self.register_buffer('coupling_coefficients', coupling_coefficients)
        self.register_buffer('coupling_coefficients_offsets', coupling_coefficients_offsets)

        self.named_buffers_pointer = partial(self.named_buffers, recurse=False)


class TensorPassingHomogenous(TensorPassingContext):
    def __init__(self, representations, radial_model, gate=torch.nn.Identity):
        super().__init__(representations)

        self.model = torch.nn.ModuleList([
            TensorPassingLayer(Rs_in, Rs_out, self.named_buffers_pointer, radial_model, gate) for (Rs_in, Rs_out) in zip(self.input_representations, self.output_representations)
        ])

    def forward(self, graph):
        features = graph.x
        for layer in self.model:
            features = layer(graph.edge_index, features, graph.abs_distances, graph.rel_vec)
        return features


class TensorPassingLayer(torch_geometric.nn.MessagePassing):
    def __init__(self, Rs_in, Rs_out, named_buffers_pointer, radial_model, gate=torch.nn.Identity):
        super().__init__()

        self.gate = gate(Rs_out)
        if not isinstance(gate, torch.nn.Identity):
            # TODO: calculate number of additional l=0 filters for gates and prepend it to Rs_out
            # implement as method in gate
            pass

        self.Rs_in = Rs_in
        self.Rs_out = Rs_out
        self.max_l = rs.lmax(Rs_in) + rs.lmax(Rs_out)

        self.register_buffer('l_out_list',      torch.tensor(rs.extract_l(Rs_out),      dtype=torch.int32))
        self.register_buffer('l_in_list',       torch.tensor(rs.extract_l(Rs_in),       dtype=torch.int32))
        self.register_buffer('mul_out_list',    torch.tensor(rs.extract_mul(Rs_out),    dtype=torch.int32))
        self.register_buffer('mul_in_list',     torch.tensor(rs.extract_mul(Rs_in),     dtype=torch.int32))

        R_base_offsets, grad_base_offsets, features_base_offsets = self._calculate_offsets()
        self.register_buffer('R_base_offsets',          torch.tensor(R_base_offsets,        dtype=torch.int32))
        self.register_buffer('grad_base_offsets',       torch.tensor(grad_base_offsets,     dtype=torch.int32))
        self.register_buffer('features_base_offsets',   torch.tensor(features_base_offsets, dtype=torch.int32))

        norm_coef = self._calculate_normalization_coefficients()
        self.register_buffer('norm_coef', norm_coef)

        n_radial_model_outputs = self.R_base_offsets[-1].item()
        self.radial_model = radial_model(n_radial_model_outputs)

        self.named_buffers_pointer = named_buffers_pointer

    def _calculate_offsets(self):
        R_base_offsets          = [0] + list(accumulate([mul_out * mul_in * (2 * min(l_out, l_in) + 1) for (mul_out, l_out) in zip(self.mul_out_list, self.l_out_list) for (mul_in, l_in) in zip(self.mul_in_list, self.l_in_list)]))
        grad_base_offsets       = [0] + list(accumulate(mul_out * (2 * l_out + 1) for (mul_out, l_out) in zip(self.mul_out_list, self.l_out_list)))
        features_base_offsets   = [0] + list(accumulate(mul_in * (2 * l_in + 1) for (mul_in, l_in) in zip(self.mul_in_list, self.l_in_list)))
        return R_base_offsets, grad_base_offsets, features_base_offsets

    def _calculate_normalization_coefficients(self):
        # TODO: allow other choice of normalization
        norm_coef = torch.empty((len(self.l_out_list), len(self.l_in_list)))
        for i, l_out in enumerate(self.l_out_list):
            num_summed_elements = sum([mul_in * (2 * min(l_out, l_in) + 1) for mul_in, l_in in zip(self.mul_in_list, self.l_in_list)])  # (l_out + l_in) - |l_out - l_in| = 2*min(l_out, l_in)
            for j, (mul_in, l_in) in enumerate(zip(self.mul_in_list, self.l_in_list)):
                norm_coef[i, j] = math.sqrt(4 * math.pi) * math.sqrt(2 * l_out + 1) / math.sqrt(num_summed_elements)

        return norm_coef

    def forward(self, edge_index, features, abs_distances, rel_vectors):
        radial_model_outputs = self.radial_model(abs_distances)
        real_spherical_harmonics = spherical_harmonics_xyz(list(range(self.max_l + 1)), rel_vectors)
        return self.propagate(edge_index, x=features, rsh=real_spherical_harmonics, rbm=radial_model_outputs)

    def message(self, x_j, rsh, rbm):
        return tensor_msg(x_j, rsh, rbm, self)

    def update(self, inputs):
        return self.gate(inputs)

    def message_and_aggregate(self, adj_t):
        raise NotImplementedError("Use separated message and aggregation")


class TensorMessageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, real_spherical_harmonics, radial_model_outputs, layer):
        F = features.transpose(0, 1).contiguous()
        Y = real_spherical_harmonics.transpose(0, 1).contiguous()
        R = radial_model_outputs.transpose(0, 1).contiguous()

        buffers = dict(layer.named_buffers_pointer())

        msg = tensor_message.forward(
            layer.norm_coef,                                    # W
            buffers['coupling_coefficients'],                   # C
            F,                                                  # F
            Y,                                                  # Y
            R,                                                  # R
            layer.l_out_list,                                   # L_out_list
            layer.l_in_list,                                    # L_in_list
            layer.mul_out_list,                                 # u_sizes
            layer.mul_in_list,                                  # v_sizes
            layer.grad_base_offsets,                            # output_base_offsets
            buffers['coupling_coefficients_offsets'],           # C_offsets
            layer.features_base_offsets,                        # F_base_offsets
            layer.R_base_offsets)                               # R_base_offsets

        msg = msg.transpose_(0, 1).contiguous()                 # [(l_out, u, i), (a, b)] -> [(a, b), (l_out, u, i)]

        if features.requires_grad or radial_model_outputs.requires_grad:
            ctx.save_for_backward(F, Y, R)                      # F, Y, R - transposed
            ctx.layer = layer

        return msg

    @staticmethod
    def backward(ctx, grad_output):
        F, Y, R = ctx.saved_tensors  # F, R - transposed
        layer = ctx.layer
        buffers = dict(layer.named_buffers_pointer())

        G = grad_output.transpose(0, 1).contiguous()

        msg_grad_F = tensor_message.backward_F(
            layer.norm_coef,                                    # W
            buffers['coupling_coefficients'],                   # C
            G,                                                  # G
            Y,                                                  # Y
            R,                                                  # R
            layer.l_out_list,                                   # L_out_list
            layer.l_in_list,                                    # L_in_list
            layer.mul_out_list,                                 # u_sizes
            layer.mul_in_list,                                  # v_sizes
            layer.features_base_offsets,                        # output_base_offsets
            buffers['coupling_coefficients_offsets'],           # C_offsets
            layer.grad_base_offsets,                            # G_base_offsets
            layer.R_base_offsets)                               # R_base_offsets

        msg_grad_F = msg_grad_F.transpose_(0, 1).contiguous()   # [(a, b), (l_in, v, j)]

        msg_grad_R = tensor_message.backward_R(
            layer.norm_coef,                                    # W
            buffers['coupling_coefficients'],                   # C
            G,                                                  # G
            F,                                                  # F
            Y,                                                  # Y
            layer.l_out_list,                                   # L_out_list
            layer.l_in_list,                                    # L_in_list
            layer.mul_out_list,                                 # u_sizes
            layer.mul_in_list,                                  # v_sizes
            layer.R_base_offsets,                               # output_base_offsets
            buffers['coupling_coefficients_offsets'],           # C_offsets
            layer.grad_base_offsets,                            # G_base_offsets
            layer.features_base_offsets)                        # F_base_offsets

        msg_grad_R = msg_grad_R.transpose_(0, 1).contiguous()   # [(a, b), (l_out, l_in, l, u, v)]

        return msg_grad_F, None, msg_grad_R, None


tensor_msg = TensorMessageFunction.apply
