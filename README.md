# Reproducible Gemma 3n QLoRA experiment

This repository fine-tunes `unsloth/gemma-3n-E4B-it` on a pinned revision of
FineTome-100k and compares the resulting QLoRA adapter with the unchanged base
model on an untouched test split. The original Kaggle notebook remains in `nb/`;
the scripts added here make its experiment reviewable and repeatable outside a
notebook.

## Experiment design

| Component | Configuration |
|---|---|
| Base model | `unsloth/gemma-3n-E4B-it` at commit `45e9fb1dd0e34db5ff9db1f43a49ac5d8e8b8778`, 4-bit loading |
| Dataset | `mlabonne/FineTome-100k` at revision `c2343c1372ff31f51aa21248db18bffa3193efdb` |
| Split | Seeded 2,400 train / 300 validation / 300 held-out test |
| QLoRA | Rank 8, alpha 8, language attention and MLP modules |
| Training | 60 steps, effective batch size 4, learning rate 2e-4 |
| Selection | Lowest validation completion loss, evaluated every 10 steps |
| Final comparison | Completion NLL/perplexity, paired quality intervals, latency, throughput, peak VRAM |

The dataset manifest stores split checksums, the source revision, dataset
fingerprint, seed, and row counts. Data preparation also fails closed when
normalized exact duplicates or high-similarity prompt shingles cross split
boundaries. The content-safe audit records only split names, source indices, and
similarity scores, never dataset text. Training never sees the test split. Both model
variants are evaluated from scratch on the same host and examples. Evaluation
fails closed if the held-out split no longer matches its manifest checksum. The
Gemma 3 chat template is applied consistently in training and evaluation, and
completion-loss tokenization left-truncates prompts while preserving every target
token. The result artifact reports the exact number of scored examples, completion
tokens, and removed prompt tokens.
The base model is loaded from a full commit SHA rather than mutable `main`. The
same revision is written into and verified from the PEFT `adapter_config.json`,
so a later adapter evaluation cannot silently resolve different base weights.
For exact match and token F1, a seeded paired bootstrap resamples the same prompts
for both variants and reports a 95% confidence interval around the QLoRA-minus-base
delta. This makes uncertainty visible without treating examples as independent
across the two model runs.

## Run the experiment

A recent NVIDIA GPU is recommended. Install the CUDA-compatible PyTorch build for
your system first, then install the experiment dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-train.txt

python scripts/prepare_data.py --config configs/experiment.json
python scripts/audit_splits.py --config configs/experiment.json
python scripts/train.py --config configs/experiment.json
python scripts/evaluate.py --config configs/experiment.json
```

Data preparation writes a versioned `leakage_audit.json` and refuses to create a
usable experiment manifest when cross-split leakage is detected. The standalone
audit command can recheck prepared files. The training script writes the adapter
and a machine-readable training summary to `artifacts/gemma-3n-qlora/`. The
evaluation script writes `results/base_vs_qlora.json`.

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
contains open-ended assistant responses. Exact match and token F1, including their
paired bootstrap intervals, are diagnostics only, not sufficient measures of
conversational quality. A serious
follow-up should add a human rubric or a documented judge model for helpfulness,
factuality, and safety.

## Reproducibility and CI

The lightweight CI job does not download the model or dataset. It checks formatting,
linting, script compilation, deterministic split behavior, data validation,
checksums, cross-split leakage detection, and metric calculations. GPU training
remains an explicit experiment,
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
- The leakage audit detects normalized and lexical near-duplicates, not semantic
  paraphrases; high-risk datasets still need embedding-based or human review.
