# GPU Phase-0 runbook (not runnable in the CPU container; verify flags against your vLLM version)

Goal (PLAN §7 P0): measure the determinism tax and D1/D2 behaviour of the hub and its dense
control on a real engine, using `experiments/gpu/p0_engine_conformance.py`.

1. **Pin the toolchain.** Record vLLM version, CUDA/driver, GPU SKU, and the HF revision of each
   model; these go into the lock (R1).
2. **Serve the hub twice** on one SKU, once in default mode and once batch-invariant
   (`VLLM_BATCH_INVARIANT=1`; per the vLLM docs this also needs a supported attention backend),
   with prefix caching, speculative decoding and async scheduling disabled (R6), e.g.
   `VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --revision <sha> --port 8001 --no-enable-prefix-caching`
   and the same without the environment variable on port 8000.
3. **D1 under load** (per endpoint):
   `python experiments/gpu/p0_engine_conformance.py --ref http://localhost:8001 --model Qwen/Qwen3-30B-A3B-Instruct-2507 --n 100`
   Gate: `d1_tar_raw == 1.0` (every golden prompt gives one unique output across 100 interleaved
   copies) on the batch-invariant endpoint. Expect < 1.0 on the default endpoint.
4. **D2-fresh across stacks:** pass `--alt` pointing at the same model on a different SKU or
   engine version (or a different TP size). Report parity at coverage C₀ = 0.85.
5. **Dense control:** repeat 2–4 with Qwen/Qwen3-32B (and Qwen/Qwen3-8B for cheap runs).
6. **Determinism tax tuple** per pair: unique@N in default mode, first divergence index, fragile
   share (needs routing traces; HF reference stack until the engine exposes routes), D2-fresh
   parity at C₀; plus the throughput ratio batch-invariant / default from the server metrics.
