# shinkansen

A training framework — fast, reliable, on time.

Pretraining-first, with post-training (SFT / RL) on the roadmap. First-class support for both **NVIDIA CUDA** and **AMD ROCm** — same code path, same configs, same throughput discipline.

## Why another framework?

Most existing options pick a side:

- **torchtitan / nanotron** — clean pretraining, but CUDA-centric.
- **axolotl / TRL** — finetuning / RL focused, not pretraining.
- **Megatron-LM** — fast but heavy, CUDA-only.

Shinkansen aims to be:

1. **Bi-architecture native** — CUDA and ROCm are both first-class. AMD-specific kernels (aiter rmsnorm/flash-attn) and NVIDIA fast paths are pluggable, not bolted on.
2. **Pretraining → post-training continuum** — the same trainer scales from from-scratch pretraining to SFT and RL post-training. No rewrite.
3. **Throughput-honest** — every config ships with measured tok/s and peak HBM on reference hardware (H100, MI300X, MI325X). No silent regressions.
4. **PyTorch-native** — FSDP2 / HSDP, `torch.compile`, no custom DSL.

## Status

🚧 Early development. Laying track.

## Roadmap

- [ ] Single-node FSDP2 pretraining loop (Llama-style)
- [ ] CUDA + ROCm parity benchmarks (tok/s, peak HBM)
- [ ] aiter kernel integration on ROCm (rmsnorm, flash-attn)
- [ ] Multi-node HSDP (RoCE / IB)
- [ ] Resumable checkpointing (DCP)
- [ ] SFT trainer
- [ ] RL trainer (GRPO-style)

## License

MIT
