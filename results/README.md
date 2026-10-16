# Experiment results

`base_vs_qlora.json` is written here by `scripts/evaluate.py`. It is intentionally
not checked in yet because this environment does not have the GPU and model access
needed to produce a genuine run.

The result artifact records:

- the SHA-256 hashes of the data manifest and untouched test split;
- the exact base-model commit and SHA-256 hash of the pinned PEFT adapter config;
- completion-only negative log-likelihood and perplexity over all 300 test examples,
  with a seeded paired, token-weighted bootstrap interval for the primary loss delta;
- paired, token-weighted completion-loss deltas across fixed original-prompt and
  completion-length slices, including the largest observed slice delta;
- exact match and token F1 as secondary reference-overlap diagnostics, with seeded
  paired-bootstrap confidence intervals for the QLoRA-minus-base deltas;
- warm-up-excluded p50/p95 greedy-generation latency, request and token throughput,
  model-load time, and steady-state peak GPU memory;
- content-free per-example loss/token records plus per-request prompt/output token
  counts and synchronized timings;
- base-to-adapter deltas, experiment configuration, and hardware/software versions.

The checked-in contract requires CUDA and excludes three warm-up requests per
variant from measured latency. Run the commands in the repository README on one
CUDA host, inspect the generated JSON, and commit that artifact only when all stages
complete successfully.
