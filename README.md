# Reproducible Gemma 3n QLoRA experiment

This repository fine-tunes `unsloth/gemma-3n-E4B-it` on a pinned revision of
FineTome-100k and compares the resulting QLoRA adapter with the unchanged base
model on an untouched test split. The original Kaggle notebook remains in `nb/`;
the scripts added here make its experiment reviewable and repeatable outside a
notebook.

## Experiment design

| Component | Configuration |
|---|---|
| Base model | `unsloth/gemma-3n-E4B-it`, 4-bit loading |
| Dataset | `mlabonne/FineTome-100k` at revision `c2343c1372ff31f51aa21248db18bffa3193efdb` |
| Split | Seeded 2,400 train / 300 validation / 300 held-out test |
| QLoRA | Rank 8, alpha 8, language attention and MLP modules |
| Training | 60 steps, effective batch size 4, learning rate 2e-4 |
| Selection | Lowest validation completion loss, evaluated every 10 steps |
| Final comparison | Completion NLL/perplexity, reference overlap, latency, throughput, peak VRAM |

The dataset manifest stores split checksums, the source revision, dataset
fingerprint, seed, and row counts. Training never sees the test split. Both model
variants are evaluated from scratch on the same host and examples. Evaluation
fails closed if the held-out split no longer matches its manifest checksum. The
Gemma 3 chat template is applied consistently in training and evaluation, and
completion-loss tokenization left-truncates prompts while preserving every target
token. The result artifact reports the exact number of scored examples, completion
tokens, and removed prompt tokens.

## Run the experiment

A recent NVIDIA GPU is recommended. Install the CUDA-compatible PyTorch build for
your system first, then install the experiment dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-train.txt

python scripts/prepare_data.py --config configs/experiment.json
python scripts/train.py --config configs/experiment.json
python scripts/evaluate.py --config configs/experiment.json
```

The training script writes the adapter and a machine-readable training summary to
`artifacts/gemma-3n-qlora/`. The evaluation script writes
`results/base_vs_qlora.json`.

## Held-out results

| Metric | Base | QLoRA | Delta |
|---|---:|---:|---:|
| Completion loss | pending GPU run | pending GPU run | pending |
| Completion perplexity | pending GPU run | pending GPU run | pending |
| Token F1 | pending GPU run | pending GPU run | pending |
| p50 generation latency | pending GPU run | pending GPU run | pending |
| p95 generation latency | pending GPU run | pending GPU run | pending |
| Generation throughput | pending GPU run | pending GPU run | pending |
| Peak GPU memory | pending GPU run | pending GPU run | pending |

No benchmark numbers are filled in without a completed real-model run. The result
JSON captures hardware and software versions so later measurements can be audited.

Completion loss and perplexity are the primary quality measures because FineTome
contains open-ended assistant responses. Exact match and token F1 are included as
diagnostics only, not as sufficient measures of conversational quality. A serious
follow-up should add a human rubric or a documented judge model for helpfulness,
factuality, and safety.

## Reproducibility and CI

The lightweight CI job does not download the model or dataset. It checks formatting,
linting, script compilation, deterministic split behavior, data validation,
checksums, and metric calculations. GPU training remains an explicit experiment,
not an unverified CI claim.

Run the local checks with:

```bash
pip install -r requirements-dev.txt
ruff format --check .
ruff check .
python -m compileall -q gemma_experiment scripts
pytest -q
```

## Limitations

- The small 3,000-row sample and 60 training steps preserve the notebook baseline;
  they are not evidence that the model is production-ready.
- A single seeded split does not quantify variance across data samples or training
  seeds.
- Base and adapter latency must be compared on identical hardware with no competing
  workload.
- FineTome is a broad instruction dataset, so downstream task-specific evaluation
  is still required before deployment.

