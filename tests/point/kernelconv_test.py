# pylint: disable=C,E1101,E1102
import unittest
from functools import partial

import torch

from e3nn.kernel import Kernel
from e3nn.radial import ConstantRadialModel
from e3nn.point.operations import Convolution
from e3nn.point.kernelconv import KernelConv
from e3nn.rs import dim


class Tests(unittest.TestCase):
    def setUp(self):
        super(Tests, self).setUp()
        torch.set_default_dtype(torch.float64)
        self.Rs_in = [(1, 0), (1, 1), (2, 0), (1, 2)]
        self.Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]

        self.geometry = torch.rand(100, 40, 3)
        self.features = torch.rand(100, 40, dim(self.Rs_in), requires_grad=True)

    def test_compare_forward_norm(self):
        for normalization in ["norm", "component"]:
            K = partial(Kernel, RadialModel=ConstantRadialModel, normalization=normalization)
            C = Convolution(K, self.Rs_in, self.Rs_out)
            new_features = C(self.features, self.geometry)

            KC = KernelConv(self.Rs_in, self.Rs_out, RadialModel=ConstantRadialModel, normalization=normalization)
            check_new_features = KC(self.features, self.geometry)

            self.assertTrue(torch.allclose(new_features, check_new_features))

    def test_compare_backward_features(self):
        check_features = self.features.clone().detach().requires_grad_()
        check_geometry = self.geometry.clone().detach()

        for normalization in ["norm", "component"]:
            K = partial(Kernel, RadialModel=ConstantRadialModel, normalization=normalization)
            C = Convolution(K, self.Rs_in, self.Rs_out)
            new_features = C(self.features, self.geometry)

            KC = KernelConv(self.Rs_in, self.Rs_out, RadialModel=ConstantRadialModel, normalization=normalization)
            check_new_features = KC(check_features, check_geometry)

            # Capture ground truth gradient
            target = torch.rand_like(new_features)
            loss = torch.norm(new_features - target)
            loss.backward()

            # Capture KernelConv gradient
            check_target = target.clone().detach()
            check_loss = torch.norm(check_new_features - check_target)
            check_loss.backward()

            self.assertTrue(torch.allclose(self.features.grad, check_features.grad))

    def _test_compare_backward_geometry(self):
        check_features = self.features.clone().detach()
        check_geometry = self.geometry.clone().detach().requires_grad_()

        for normalization in ["norm", "component"]:
            K = partial(Kernel, RadialModel=ConstantRadialModel, normalization=normalization)
            C = Convolution(K, self.Rs_in, self.Rs_out)
            new_features = C(self.features, self.geometry)

            KC = KernelConv(self.Rs_in, self.Rs_out, RadialModel=ConstantRadialModel, normalization=normalization)
            check_new_features = KC(check_features, check_geometry)

            # Capture ground truth gradient
            target = torch.rand_like(new_features)
            loss = torch.norm(new_features - target)
            loss.backward()

            # Capture KernelConv gradient
            check_target = target.clone().detach()
            check_loss = torch.norm(check_new_features - check_target)
            check_loss.backward()

            self.assertTrue(torch.allclose(self.geometry.grad, check_geometry.grad))


if __name__ == '__main__':
    unittest.main()
