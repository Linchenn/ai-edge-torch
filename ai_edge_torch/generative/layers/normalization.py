# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Common normalization layers.

import torch
import torch.nn.functional as F
from torch import nn
from ai_edge_torch.hlfb import StableHLOCompositeBuilder


# Implementation for RMSNorm from: https://arxiv.org/abs/1910.07467
class RMSNorm(torch.nn.Module):

  def __init__(self, dim: int, eps: float = 1e-6, zero_centered_gamma=False):
    """Initialize the RMSNorm layer.

    Args:
      dim (int): dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default:
        1e-6).
    """
    super().__init__()
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.ones(dim))
    self.zero_centered_gamma = zero_centered_gamma

  def _norm(self, x):
    """Apply RMSNorm normalization.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: The normalized output tensor.
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    """Running the forward pass of RMSNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying RMSNorm.
    """
    output = self._norm(x.float()).type_as(x)
    if self.zero_centered_gamma:
      return output * (1 + self.weight)
    else:
      return output * self.weight


class GroupNorm(torch.nn.Module):

  def __init__(self, group_num: int, dim: int, eps: float = 1e-6, enable_hlfb: bool = False):
    """Initialize the RMSNorm layer.

    Args:
      dim (int): dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default:
        1e-6).
    """
    super().__init__()
    self.enable_hlfb = enable_hlfb
    self.group_num = group_num
    self.eps = eps
    self.norm = nn.GroupNorm(group_num, dim, eps)

  def forward(self, x):
    """Running the forward pass of RMSNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying RMSNorm.
    """
    if self.enable_hlfb:
      return group_norm_with_hlfb(
          x,
          self.norm.weight,
          self.norm.bias,
          self.group_num,
          self.eps,
      )
    else:
      return self.norm(x)


class LayerNorm(torch.nn.Module):

  def __init__(self, dim: int, eps: float = 1e-6, enable_hlfb: bool = False):
    """Initialize the RMSNorm layer.

    Args:
      dim (int): dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default:
        1e-6).
    """
    super().__init__()
    self.enable_hlfb = enable_hlfb
    self.eps = eps
    self.norm = nn.LayerNorm(dim, eps=eps)

  def forward(self, x):
    """Running the forward pass of RMSNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying RMSNorm.
    """
    if self.enable_hlfb:
      return layer_norm_with_hlfb(
          x,
          self.norm.weight,
          self.norm.bias,
          self.eps,
      )
    else:
      return self.norm(x)

def group_norm_with_hlfb(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    num_groups: int,
    eps: float,
):
  x = torch.permute(x, (0, 2, 3, 1))

  builder = StableHLOCompositeBuilder(
      name="odml.group_norm", attr={"num_groups": num_groups, "eps": eps}
  )
  x, w, b = builder.mark_inputs(x, w, b)
  x = torch.permute(x, (0, 3, 1, 2))
  y = F.group_norm(x, num_groups, weight=w, bias=b, eps=eps)
  y = torch.permute(y, (0, 2, 3, 1))
  y = builder.mark_outputs(y)

  y = torch.permute(y, (0, 3, 1, 2))
  return y

def layer_norm_with_hlfb(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float,
):
  builder = StableHLOCompositeBuilder(
      name="odml.layer_norm", attr={"eps": eps}
  )
  x, w, b = builder.mark_inputs(x, w, b)
  y = F.layer_norm(x, x.shape, weight=w.broadcast_to(x.shape), bias=b.broadcast_to(x.shape), eps=eps)
  y = builder.mark_outputs(y)
  return y
