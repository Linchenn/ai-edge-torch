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

import ai_edge_torch
from ai_edge_torch.convert import conversion_utils as cutils
from ai_edge_torch.testing import model_coverage
from ai_edge_torch.generative.layers.normalization import GroupNorm  # NOQA


class demo(unittest.TestCase):
  class MyGroupNorm(torch.nn.Module):

    def forward(self, q, k, v, mask):
      return GroupNorm(q, k, v, 40, mask)

  args = (
      torch.randn((2, 4, 8, 16)),
  )
  torch_module = GroupNorm(16, 16, 1e-5, True).eval()
  src = torch.randn((2, 4, 8, 16))
  out = torch_module(src)
  edge_model = ai_edge_torch.convert(torch_module, args).export('/tmp/groupNorm.tflite')


if __name__ == "__main__":
  demo()
