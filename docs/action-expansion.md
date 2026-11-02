# H100 action-set comparison

The H100 comparison measures how ten additional Triton configurations change the gap to `torch.matmul`, while the original 36 configurations stay exactly as they were.

## Fixed action set

I add four ordinary tensor-core tiles `(BM, BN, BK, warps, stages)`: `(128,128,64,4,5)`, `(128,256,64,8,4)`, `(256,128,64,8,4)` and `(128,128,128,8,3)`. I also use the existing persistent TMA implementation with configurations `(128,128,64,4,4)` and `(128,256,64,8,3)`, both without warp specialization or a subtiled epilogue, and these six candidates apply to every workload.

For M=1,7,31 I add four tensor-core split-K configurations with BM=16, BK=64, four warps, three stages, BN in {64,128} and split factor in {4,16}. Each one writes FP32 partial tiles to a workspace and then runs a separate Triton FP32 reduction that converts once to FP16, and because allocating both buffers and executing both kernels are part of the timed call, I never time an isolated partial kernel.

The CPU manifest in `src/quartet/action_configs.py` defines the ten new actions, and `src/quartet/action_expansion.py` launches them without fallback. I use Torch 2.8.0 and Triton 3.4.0, whose persistent tensor-descriptor implementation is already present in this project and is based on [Triton's pinned persistent-matmul tutorial](https://github.com/triton-lang/triton/blob/v3.4.0/python/tutorials/09-persistent-matmul.py).

## Selection and measurement

The suite has 96 named workloads and 84 distinct (M,N,K) shapes. Pass 1 measures all 36 original configurations and every applicable new configuration, and for each workload it selects the lowest correct median separately within the old and new sets, breaking ties by configuration key. Pass 2 then measures those two fixed winners and `torch.matmul`, while the combined set chooses between old and new using the pass-1 medians. The JSON files call these passes banks 1 and 2.

Both banks use uniform FP16 operands, and the workload order is shuffled with the bank number. The tensor seed is bank*10000 plus the index in that shuffled order, which matches the collector's original default seed convention, and candidate order uses bank*10000+index+70000 so that the larger cross-product also has a recorded deterministic shuffle. The reference is `torch.mm` with FP32 output and TF32 disabled only for that reference call and restored afterwards, and every candidate has to pass the unchanged elementwise atol=rtol=0.01 comparison and a finite-output check before it is timed. Failed compilation, correctness and timing attempts remain in the raw data, so no configuration is swapped in for them after bank 2.

Timing follows the original protocol of allocating CUDA FP16 calls, Triton `do_bench`, a 25 ms warmup, a 100 ms repetition interval and p20/median/p80. I measure `torch.matmul` in both banks in the same session and record the precision flags, GPU identity and software versions, which is also why I never compare current times directly with earlier measurements from another session.

## Primary analysis

For each workload the ratio is the pass-2 median of the selected new action divided by the torch median from the same pass, so a value below one is a Triton win. The summary includes old, new and combined-set win counts, per-workload ratios and selected configurations, geometric means, M-group summaries and failures.

The pilot measures four large-K workloads at M=1,7,31,1024 and is saved separately, and it estimates compilation and timing cost without changing the action set, the timing windows or the numerical tolerance. The full run then repeats every workload and may reuse compiled binaries, while pilot timings are excluded from the reported result.

Raw measurements are in [`results/action-expansion-raw.json`](../results/action-expansion-raw.json), and [`scripts/analyze_action_expansion.py`](../scripts/analyze_action_expansion.py) builds the summary and the per-workload figure. Each collection records the source commit and the latest commit to this method document.
