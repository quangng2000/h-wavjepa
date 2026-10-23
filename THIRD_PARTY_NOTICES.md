# Third-party notices

The root [MIT license](LICENSE) covers this project's original code. The notices
below retain the licenses for imported and adapted portions; the root license
does not relicense external datasets, pretrained weights, or dependencies.

## WavJEPA

The actual core from [labhamlet/wavjepa](https://github.com/labhamlet/wavjepa),
commit `6f4e1b1f18316c50768c8523f325ccffeaa6985b`, is retained in
`src/vendor/wavjepa`. See its [source/diff record](src/vendor/wavjepa/UPSTREAM.json),
[notices](src/vendor/wavjepa/NOTICE.md), [BSD license](src/vendor/wavjepa/LICENSE),
and [historical AudioMAE license](src/vendor/wavjepa/LICENSE-AudioMAE).
The copied AudioMAE positional utilities have been removed from active code and
replaced with an independent implementation of the sinusoidal formula. The
original source hash and license remain for historical provenance; no broader
commercial-rights claim is made for pretrained weights or dependencies.
Unrelated upstream denoising, data-loading and evaluation vendor packages are
not imported. Source reuse does not imply upstream endorsement or reproduced results.

## H-JEPA SIGReg

`src/objectives/sigreg.py` adapts SIGReg from `h_jepa/loss.py` in
[kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA), commit
`97054ecaff83e13839195ec4b92d74c08a0127c5`. The single-device formula is retained;
validation and an autograd-aware global-population DDP adapter are added.
The original upstream distributed reduction is not used. CPU multi-process tests
compare the adapter's loss and parameter gradients with one concatenated batch;
statistics remain FP32 during BF16 mixed-precision training.
The [MIT license and author notice](third_party/licenses/H-JEPA.txt) are retained.
Default projection count is configurable and differs from the paper's recipe.
The unmodified `tests/fixtures/h_jepa/loss.py` at the same revision is retained
solely for independent numerical and gradient comparisons; its SHA-256 is recorded
alongside it. The same MIT notice applies to that fixture.

## LocaLeJEPA LocaLeReg

`src/vendor/localereg/losses.py` is an unmodified copy of `src/losses.py` from
[alessioborgi/LocaLeJEPA](https://github.com/alessioborgi/LocaLeJEPA), commit
`771361c18d37e65df3dc188761ffb26751ce4e37`. The [MIT license and copyright
notice](third_party/licenses/LocaLeJEPA.txt) are retained. The [manifest](src/vendor/localereg/UPSTREAM.json)
records its Git blob and SHA-256 hashes, verified by a test. The audio adapter
calls its built-in torch k-means backend; optional upstream packages are not needed.
[Adaptations](docs/LOCALEREG.md) include independent per-time neighborhoods and
a higher quadrature resolution for 768-dimensional latents. Reuse does not
reproduce the paper's training experiment or establish audio benchmark gains.

## HEAR evaluation kit

The optional isolated evaluator installs [hearbenchmark/hear-eval-kit](https://github.com/hearbenchmark/hear-eval-kit)
at commit `855964977238e89dfc76394aa11c37010edb6f20` under its Apache-2.0 license.
Its code is installed as a separate dependency, not bundled in this repository.
The SHA-256 guarded compatibility patches make TensorFlow optional and adapt the
chosen-epoch lookup for Lightning 1.9 event postprocessing. They preserve upstream
probe models, label alignment, metrics and validation selection. See
[evaluation setup and patch details](docs/HEAR.md).

## Lightning-Hydra-Template

This project imports and adapts code, configurations, and development workflow files from [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), revision `bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144`. See [the integration record](docs/TEMPLATE.md) for the imported file scope and project-specific changes.

The following notice is reproduced from the upstream README and applies to those imported and adapted portions.

```text
MIT License

Copyright (c) 2021 ashleve

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
