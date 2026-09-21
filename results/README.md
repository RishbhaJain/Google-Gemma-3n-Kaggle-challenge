# Experiment results

`base_vs_qlora.json` is written here by `scripts/evaluate.py`. It is intentionally
not checked in yet because this environment does not have the GPU and model access
needed to produce a genuine run.

The result artifact records:

- the SHA-256 hashes of the data manifest and untouched test split;
- completion-only negative log-likelihood and perplexity over all 300 test examples;
- exact match and token F1 as secondary reference-overlap diagnostics;
- p50/p95 greedy-generation latency, throughput, and peak GPU memory;
- base-to-adapter deltas, experiment configuration, and hardware/software versions.

Run the three commands in the repository README on one CUDA host, inspect the
generated JSON, and commit that artifact only when all stages complete successfully.

