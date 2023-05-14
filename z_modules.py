import torch
from torch import nn
import numpy as np
from torch.autograd import Function
from typing import Any, NewType

# --------------------------------------------
# Binarized module by Straight-through estimator (STE)
# --------------------------------------------

def binary_sign(x: torch.Tensor):
    """Return -1 if x < 0, 1 if x >= 0."""
    return x.sign() + (x == 0).type(torch.float)  # type: ignore

class STESign(Function):
    """
    Binarize tensor using sign function.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):  # type: ignore
        """
        Return a Sign tensor.
        Args:
            ctx: context
            x: input tensor
        Returns:
            Sign(x) = (x>=0) - (x<0)
            Output type is float tensor where each element is either -1 or 1.
        """
        ctx.save_for_backward(x)
        sign_x = binary_sign(x)
        return sign_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore  # pragma: no cover (since this is called by C++ code) # noqa: E501
        """
        Compute gradient using STE.
        Args:
            ctx: context
            grad_output: gradient w.r.t. output of Sign
        Returns:
            Gradient w.r.t. input of the Sign function
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.gt(1)] = 0
        grad_input[x.lt(-1)] = 0
        return grad_input


# Convenience function to binarize tensors
STESign_fc = STESign.apply    # results in -1|1
def STEBinary_fc(x):
    return (STESign_fc(x)+1)/2  # results in 0|1


# --------------------------------------------
# Coded blur generation  
# --------------------------------------------
class CEBlurNet(nn.Module):
    '''
    optimized motion blur encoder: using optimized coded exposure sequence to generate optimized motion blur for input image.
    - weight binarize: STE-LBSIGN
    '''

    def __init__(self, sigma_range=0, test_sigma_range=0, ce_code_n=8, frame_n=8, ce_code_init=None, opt_cecode=False):
        super(CEBlurNet, self).__init__()
        # self.device = device
        self.sigma_range = sigma_range
        # sigma range for validation and final test
        self.test_sigma_range = test_sigma_range
        self.frame_n = frame_n  # frame num
        self.time_idx = torch.linspace(
            0, 1, ce_code_n).unsqueeze(0).t()  # time idx
        self.upsample_factor = frame_n//ce_code_n
        self.binary_fc = STEBinary_fc
        self.ce_code_n = ce_code_n
        self.ce_weight = nn.Parameter(torch.Tensor(ce_code_n, 1))
        if ce_code_init is None:
            nn.init.uniform_(self.ce_weight, a=-1, b=1)  # initialize
            # self.weight.data = torch.randint(0, 2, (ce_code_n, 1)).float()
        else:
            # convert 0,1 (ce_code) -> -1,1 (ce_weight)
            ce_code_init = [-1 if ce_code_init[k] ==
                            0 else 1 for k in range(len(ce_code_init))]
            self.ce_weight.data = torch.tensor(
                ce_code_init, dtype=torch.float32).unsqueeze(0).t()
        if not opt_cecode:
            # whether optimize ce code
            self.ce_weight.requires_grad = False

        # upsample matrix for ce_code(parameters)
        self.upsample_matrix = torch.zeros(
            self.upsample_factor * ce_code_n, ce_code_n)
        for k in range(ce_code_n):
            self.upsample_matrix[k *
                                 self.upsample_factor:(k+1)*self.upsample_factor, k] = 1

    def forward(self, frames):
        # info
        device = frames.device
        # binarize and upsample ce_code (weights)
        ce_code = self.binary_fc(self.ce_weight)  # weights binarized
        ce_code_up = torch.matmul(
            self.upsample_matrix.to(device), ce_code)
        assert ce_code_up.data.shape[0] == frames.shape[
            1], f'frame num({frames.shape[1]}) is not equal to CeCode length({ce_code_up.shape[0]})'

        # frames to blur encoding
        ce_code_up_ = ce_code_up.view(self.frame_n, 1, 1, 1).expand_as(frames)
        ce_blur_img = torch.sum(
            ce_code_up_*frames, axis=1)/self.frame_n

        # add noise
        sigma_range = self.sigma_range if self.training else self.test_sigma_range
        if isinstance(sigma_range, (int, float)):
            noise_level = sigma_range
        else:
            noise_level = np.random.uniform(*sigma_range)

        ce_blur_img_noisy = ce_blur_img + torch.tensor(noise_level, device=device) * \
            torch.randn(ce_blur_img.shape, device=device)

        return ce_blur_img_noisy, self.time_idx.to(device), ce_code, ce_blur_img # zzh: ce_code_up->ce_code

