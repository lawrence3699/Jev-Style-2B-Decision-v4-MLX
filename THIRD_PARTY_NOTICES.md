# Third-party notices

This file covers third-party code that is **contained in** this repository. Libraries that are only imported at
run time (mlx, mlx-lm, tokenizers, numpy) are not redistributed here and keep their own licences.

## mlx-lm 0.31.3 (MIT License)

`jev_style_decision_mlx.py` contains a modified copy of `GatedDeltaNet.__call__` from
`mlx_lm/models/qwen3_5.py` of mlx-lm 0.31.3 (https://github.com/ml-explore/mlx-lm; that source file carries the
header "Copyright © 2026 Apple Inc."). In the runtime it is the function `_gdn_hf_call`. Changes made:

- the q/k normalisation epsilon: `mx.fast.rms_norm(x, None, 1e-6)` became `mx.fast.rms_norm(x, None, 1e-6 / Dk)`,
  so that q/k are l2-normalised with eps 1e-6 as in Hugging Face transformers / llama.cpp;
- the tensor-parallel (sharding) branches were removed: a sharded layer is refused with an error;
- rewritten as a module-level function (installed per model instance, never globally) and reformatted.

The mlx-lm licence (the LICENSE file distributed with mlx-lm 0.31.3), reproduced verbatim:

```text
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
