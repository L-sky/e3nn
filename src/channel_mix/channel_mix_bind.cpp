#include <torch/extension.h>

void forward_cuda(
        torch::Tensor output,
        torch::Tensor W,
        torch::Tensor F,
        torch::Tensor L_list,
        torch::Tensor v_sizes,
        torch::Tensor w_sizes,
        torch::Tensor norm);

void backward_F_cuda(
        torch::Tensor output,
        torch::Tensor W,
        torch::Tensor G,
        torch::Tensor L_list,
        torch::Tensor v_sizes,
        torch::Tensor w_sizes,
        torch::Tensor norm);

void backward_W_cuda(
        torch::Tensor output,
        torch::Tensor G,
        torch::Tensor F,
        torch::Tensor L_list,
        torch::Tensor v_sizes,
        torch::Tensor w_sizes,
        torch::Tensor norm);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT_DTYPE(x) TORCH_CHECK(x.dtype() == torch::kFloat64 || x.dtype() == torch::kFloat32, #x " must be either float32 or float64")
#define CHECK_INT_DTYPE(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be int32")

#define CHECK_FLOAT_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT_DTYPE(x);
#define CHECK_INT_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT_DTYPE(x);


torch::Tensor forward(
        torch::Tensor W,            // [k(L) (w, v)]
        torch::Tensor F,            // [k(L) (vm), a]
        torch::Tensor L_list,
        torch::Tensor v_sizes,
        torch::Tensor w_sizes
){
    CHECK_FLOAT_INPUT(W);
    CHECK_FLOAT_INPUT(F);
    CHECK_INT_INPUT(L_list);
    CHECK_INT_INPUT(v_sizes);
    CHECK_INT_INPUT(w_sizes);

    const uint32_t a_size = (uint32_t) F.size(1);
    uint32_t lwm_size = 0;
    for (uint32_t idx = 0; idx < L_list.size(0); idx++){
        lwm_size += (2 * (uint32_t) L_list[idx].item<int32_t>() + 1) * (uint32_t) w_sizes[idx].item<int32_t>();
    }

    torch::Tensor norm = torch::rsqrt(v_sizes.to(F.dtype()));

    torch::Tensor output = torch::zeros({lwm_size, a_size}, F.options());
    forward_cuda(output, W, F, L_list, v_sizes, w_sizes, norm);
    return output;
}


torch::Tensor backward_F(
        torch::Tensor W,            // [k(L) (w, v)]
        torch::Tensor G,            // [k(L) (wm), a]
        torch::Tensor L_list,
        torch::Tensor v_sizes,
        torch::Tensor w_sizes
){
    CHECK_FLOAT_INPUT(W);
    CHECK_FLOAT_INPUT(G);
    CHECK_INT_INPUT(L_list);
    CHECK_INT_INPUT(v_sizes);
    CHECK_INT_INPUT(w_sizes);

    const uint32_t a_size = (uint32_t) G.size(1);
    uint32_t lvm_size = 0;
    for (uint32_t idx = 0; idx < L_list.size(0); idx++){
        lvm_size += (2 * (uint32_t) L_list[idx].item<int32_t>() + 1) * (uint32_t) v_sizes[idx].item<int32_t>();
    }

    torch::Tensor norm = torch::rsqrt(v_sizes.to(G.dtype()));

    torch::Tensor output = torch::zeros({lvm_size, a_size}, G.options());
    backward_F_cuda(output, W, G, L_list, v_sizes, w_sizes, norm);
    return output;
}


torch::Tensor backward_W(
        torch::Tensor G,            // [k(L) (wm), a]
        torch::Tensor F,            // [k(L) (vm), a]
        torch::Tensor L_list,
        torch::Tensor v_sizes,
        torch::Tensor w_sizes
){
    CHECK_FLOAT_INPUT(G);
    CHECK_FLOAT_INPUT(F);
    CHECK_INT_INPUT(L_list);
    CHECK_INT_INPUT(v_sizes);
    CHECK_INT_INPUT(w_sizes);

    uint32_t wv_size = 0;
    for (uint32_t idx = 0; idx < L_list.size(0); idx++){
        wv_size += (uint32_t) w_sizes[idx].item<int32_t>() * (uint32_t) v_sizes[idx].item<int32_t>();
    }

    torch::Tensor norm = torch::rsqrt(v_sizes.to(F.dtype()));

    torch::Tensor output = torch::zeros({wv_size}, G.options());
    backward_W_cuda(output, G, F, L_list, v_sizes, w_sizes, norm);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Channel mixing for irrep tensors: forward pass (CUDA)");
  m.def("backward_F", &backward_F, "Channel mixing for irrep tensors: backward pass for features (CUDA)");
  m.def("backward_W", &backward_W, "Channel mixing for irrep tensors: backward pass for weights (CUDA)");
}