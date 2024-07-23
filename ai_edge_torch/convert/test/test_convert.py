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


import os
import tempfile
import unittest

import torch
import torchvision

import ai_edge_torch
from ai_edge_torch.convert import conversion_utils as cutils
from ai_edge_torch.testing import model_coverage
from ai_edge_torch.generative.layers.scaled_dot_product_attention import scaled_dot_product_attention_with_hlfb  # NOQA


class demo(unittest.TestCase):
  class MySdpa(torch.nn.Module):

    def forward(self, q, k, v, mask):
      return scaled_dot_product_attention_with_hlfb(q, k, v, 4, mask)

  args = (
      torch.randn((1, 4, 8, 4)),
      torch.randn((1, 4, 8, 4)),
      torch.randn((1, 4, 8, 4)),
      torch.randn((1, 1, 4, 4)),
  )
  torch_module = MySdpa().eval()
  q = torch.randn((1, 4, 8, 4))
  k = torch.randn((1, 4, 8, 4))
  v = torch.randn((1, 4, 8, 4))
  mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
  out = torch_module(q, k, v, mask)
  print(out)
  edge_model = ai_edge_torch.convert(torch_module, args).export('/tmp/stable_diffusion/sdpa.tflite')


if __name__ == "__main__":
  demo()
