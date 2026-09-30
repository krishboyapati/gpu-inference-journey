# gpu-inference-journey

A 9-month deep dive into **how LLMs actually run on GPUs**, starting from a data engineering background. The question driving it: what happens between sending a prompt and getting tokens back, and why is it fast or slow?

The path goes from the serving engine (vLLM, SGLang) down to the kernels and the hardware limits underneath. This repo is where I keep the notes, kernels, benchmarks, and write-ups along the way.

- **Started:** Sep 30, 2026
- **Pace:** ~12 hrs/week alongside a full-time job, ~470 hrs total
- **Compute:** ~$260–500 in rented GPU time

---

## Contents

- [Plan at a glance](#plan-at-a-glance)
- [Phase 0: Foundations](#phase-0-foundations-weeks-16-70-hrs)
- [Phase 1: CUDA, Triton, profiling](#phase-1-cuda-triton-profiling-weeks-716-120-hrs)
- [Phase 2: Inference systems](#phase-2-inference-systems-weeks-1730-170-hrs)
- [Phase 3: Contribute and share](#phase-3-contribute-and-share-weeks-3139-110-hrs)
- [Weekly schedule and compute budget](#weekly-schedule-and-compute-budget)
- [Risks and what was left out](#risks-and-what-was-left-out)
- [Repo layout](#repo-layout)
- [Sources](#sources)

---

## Plan at a glance

Don't enter a phase until the previous one's exit criteria are met. Each phase builds on the one before: hardware intuition → kernels → serving systems → contributing back.

| Phase | Weeks | Hours | Focus | Key deliverable | GPU cost |
|---|---|---|---|---|---|
| 0 | 1–6 | ~70 | C++ reading, GPU architecture, roofline thinking | vLLM served + benchmarked; first CUDA kernels | ~$5 |
| 1 | 7–16 | ~120 | CUDA, Triton, Nsight profiling | Fused RMSNorm/SwiGLU Triton kernel repo | $45–90 |
| 2 | 17–30 | ~170 | vLLM/SGLang internals, quantization, parallelism, benchmarking | MoE cost/latency study + first upstream PR | $180–300 |
| 3 | 31–39 | ~110 | Upstream PRs, a competition, writing | 3+ merged PRs, 3 write-ups | $30–100 |

```
Wk  1 ──────── 6 ─────────────── 16 ──────────────────────── 30 ─────────────── 39
    │ Phase 0   │ Phase 1          │ Phase 2                    │ Phase 3          │
    │ Foundations│ CUDA/Triton/prof │ Inference systems          │ Contribute/share │
                                        ▲ first PR starts wk 22
```

---

## Phase 0: Foundations (weeks 1–6, ~70 hrs)

**Goal:** read C++/CUDA comfortably, reason about GPUs with numbers, and have vLLM running in week 1 so the target stays concrete.

| Block | Resource | Hours |
|---|---|---|
| Week 1 hook: serve a small model | Rent a 4090, follow the [vLLM quickstart](https://docs.vllm.ai/), run `vllm bench serve` against it, record tokens/sec and time-to-first-token | 4 |
| Linux / SSH / Docker / nvidia-smi | Gaps only; VSCode Remote-SSH into the pod | 5 |
| C++ to reading level | [learncpp.com](https://www.learncpp.com/): pointers, references, memory, classes, template basics. Skip the rest | 20 |
| GPU vocabulary | [Modal GPU Glossary](https://modal.com/gpu-glossary) | 5 |
| GPU architecture + first kernels | *Programming Massively Parallel Processors* (PMPP, 4th ed.), ch. 1–6, with [GPU MODE](https://github.com/gpu-mode/lectures) lectures 1–4 | 26 |
| Performance thinking | Horace He, ["Making Deep Learning Go Brrrr"](https://horace.io/brrr_intro.html); [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): rooflines + transformer inference chapters | 10 |

**Exit criteria:**
- [ ] vLLM served and benchmarked on a rented GPU, numbers written down
- [ ] Can compute the arithmetic intensity of a GEMM and say whether it's compute- or memory-bound on a given GPU
- [ ] Can explain, with numbers, why decode is memory-bound and prefill is compute-bound
- [ ] Wrote, compiled and launched a CUDA vector-add and a naive matmul

**Compute cost:** ~$5 (free Colab/Kaggle T4 covers most of it).

---

## Phase 1: CUDA, Triton, profiling (weeks 7–16, ~120 hrs)

**Goal:** write correct kernels, profile them, and state how close they get to hardware limits. The measurement discipline matters more than the kernel itself.

| Block | Resource | Hours |
|---|---|---|
| Matmul optimization progression | Simon Boehm, ["How to Optimize a CUDA Matmul Kernel"](https://siboehm.com/articles/22/CUDA-MMM): implement kernels 1–6 yourself, don't copy | 30 |
| Profiling | Nsight Compute + Nsight Systems on every kernel; GPU MODE lectures 8 (performance checklist) and 16 (hands-on profiling) | 10 |
| Reductions + softmax | GPU MODE lecture 9; PMPP reduction chapter | 8 |
| Triton | Official [Triton tutorials](https://triton-lang.org/) (vector add → fused softmax → matmul → layernorm → fused attention); [Triton-Puzzles](https://github.com/srush/Triton-Puzzles); GPU MODE lecture 14 | 27 |
| Attention as a kernel problem | GPU MODE lecture 12 (FlashAttention); FlashAttention paper, algorithm sections only | 8 |
| Plugging kernels into PyTorch | GPU MODE lecture 1 (profiling + integrating kernels in PyTorch); torch.compile basics | 7 |
| Deliverable | See below | 30 |

**Deliverable: fused residual-add + RMSNorm (or SwiGLU) Triton kernel repo**
- Correctness tests across fp16/bf16 and realistic LLM shapes (hidden 4096–8192, batch 1–256)
- Benchmarks vs PyTorch eager and `torch.compile`, reported as **% of peak memory bandwidth**
- A 1-page README: what's memory-bound, what you fused, why it's faster or why it isn't. Losing to `torch.compile` and explaining why is still a good result.

**Exit criteria:**
- [ ] Deliverable repo public on GitHub
- [ ] Can read an Nsight Compute report and name the bottleneck (bandwidth, occupancy, bank conflicts, stalls)
- [ ] Can explain tiling, shared memory, coalescing and tensor-core use in the matmul progression

**Compute cost:** ~$45–90 (4090 at ~10 hrs/week).

---

## Phase 2: Inference systems (weeks 17–30, ~170 hrs)

**Goal:** understand a real serving engine from the inside: how requests get scheduled and batched, how the KV cache is managed, and what quantization and parallelism actually buy you.

| Block | Resource | Hours |
|---|---|---|
| vLLM architecture | *Inside vLLM: Anatomy of a High-Throughput LLM Inference System* ([vLLM blog](https://blog.vllm.ai/)): scheduler, paged attention, continuous batching, prefix caching, spec decoding, P/D disaggregation, multi-GPU | 6 |
| vLLM source, hands-on | [vLLM repo](https://github.com/vllm-project/vllm): trace one request end to end in a debugger (API server → scheduler → KV cache manager → model runner → sampler) | 20 |
| SGLang | [SGLang](https://github.com/sgl-project/sglang) docs + GPU MODE lecture 35 (SGLang performance); RadixAttention prefix caching | 8 |
| Speculative decoding | GPU MODE lecture 22 (spec decoding in vLLM) | 4 |
| Quantization | vLLM quantization docs: FP8, AWQ/GPTQ INT4, MXFP4/NVFP4; accuracy vs throughput tradeoffs; quantize a model yourself with [llm-compressor](https://github.com/vllm-project/llm-compressor) | 15 |
| Parallelism | How to Scale Your Model sharding + inference chapters; vLLM distributed serving docs; tensor, pipeline, expert parallelism | 12 |
| CUDA graphs + torch.compile in serving | vLLM docs on CUDA graph capture and compilation | 6 |
| Collectives | GPU MODE lecture 17 (NCCL) | 3 |
| Benchmark methodology | `vllm bench serve`; p50/p99 TTFT and inter-token latency; goodput under an SLO; realistic request-length distributions | 6 |
| Reading PRs, issues, design docs | vLLM/SGLang GitHub, weekly | 15 |
| Production ops | Serve vLLM behind Kubernetes with request-level metrics, autoscaling, observability | ~10 |
| Deliverable A | Cost/latency study, below | 45 |
| Deliverable B | First upstream PR, below | 30 |

**Deliverable A: "Self-hosting an open MoE model: what it actually costs."**
- Pick a current open-weight MoE that fits on 1–4 H100s (check what's trending on Hugging Face at the time)
- Sweep: BF16 vs FP8 vs INT4; concurrency 1–256; TP=1/2/4; speculative decoding on/off; prefix caching on/off
- Report **cost per 1M tokens at a p99 latency target**, not peak throughput. Compare against API pricing and a managed option like Databricks serving.
- Strongest version: run it as an approved work project on a real internal use case, so the numbers answer a real question.

**Deliverable B: first merged PR to vLLM or SGLang (start by week 22)**
- Start with `good first issue` labels, benchmark fixes, test coverage, docs for features you've actually used
- Then a real bug fix you hit during Deliverable A. That's the most natural route to a meaningful PR.

**Exit criteria:**
- [ ] Can whiteboard a request's path through vLLM and explain the KV-cache block manager
- [ ] Deliverable A published (GitHub + blog post)
- [ ] At least one PR merged upstream, a second one open

**Compute cost:** ~$180–300 (4090 for iteration, ~10 H100-hrs/month, one 4×H100 session/month).

---

## Phase 3: Contribute and share (weeks 31–39, ~110 hrs)

**Goal:** give back to the projects I learned from and write up what I found. Explaining something publicly is the best test of whether I understand it.

| Block | What | Hours |
|---|---|---|
| More upstream PRs | Two more merged, at least one performance-related (a benchmark result in the PR description) | 40 |
| Competition entry | Whatever is live on GPU MODE (Discord → competitions channel). Submissions run on their hardware, so it's free datacenter GPU time | 30 |
| Writing | 2 posts beyond Deliverable A: "tracing a request through vLLM" and the Phase 1 kernel write-up | 15 |
| Self-check: can I explain it? | See below | 20 |
| Community | GPU MODE Discord and vLLM community channels | 5 + ongoing |

### Self-check: can I explain it?

- **Design a deployment:** serve model X at N requests/sec under a p99 latency target for $Y/month. Pick GPU count, quantization, parallelism, batching. Deliverable A is the practice run.
- **Back-of-envelope math:** KV-cache memory per token, max concurrent sequences per GPU, decode tokens/sec from memory bandwidth. Do these without notes. The core KV-cache formula:

  $$\text{KV bytes per token} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes per element}$$

- **Kernels from scratch:** reduction, softmax, tiled matmul, explained while writing
- **A real debugging story:** one performance problem found with a profiler and fixed, with numbers

**Exit criteria:**
- [ ] 3+ merged upstream PRs
- [ ] 3 public write-ups
- [ ] Can do inference sizing math on a whiteboard in under 10 minutes

---

## Weekly schedule and compute budget

The plan assumes **12 hrs/week** alongside a full-time job. Below ~8 hrs/week it stretches past 14 months and momentum usually dies. If that happens, cut scope, not quality: drop Phase 1's attention block and the competition entry.

| Slot | Hours | What goes here |
|---|---|---|
| Mon–Thu, 1 hr/night | 4 | Reading, lectures, source code. No GPU needed |
| Saturday | 4 | GPU session: build, benchmark, profile |
| Sunday | 4 | GPU session or write-up; **commit to GitHub every Sunday** |

**Compute budget (late-September 2026 rates):**

| Phase | GPUs | Estimated cost |
|---|---|---|
| 0 | Free Colab/Kaggle T4, one 4090 session | ~$5 |
| 1 | RTX 4090, ~$0.34/hr on RunPod Community Cloud | $45–90 |
| 2 | 4090 + H100 ($1.49–$3.49/hr) + monthly 4×H100 session | $180–300 |
| 3 | Mostly free competition hardware + occasional H100 | $30–100 |
| **Total** | | **~$260–500** |

**Rules:** prepaid credit only · stop pods when you stand up · keep model weights on a persistent volume · never use hyperscaler on-demand GPUs for this.

---

## Risks and what was left out

The plan is weakest on timeline and production operations.

### Weak spots

1. **The timeline assumes zero missed weeks.** Budget 20–30% slack: plan for 9 months, expect 11–12. Protect the Phase 2 deliverables first if you slip.
2. **Upstream PRs depend on other people.** vLLM/SGLang review queues can take weeks, which is why the first PR starts at week 22, not week 35. Keep PRs small and tied to bugs you actually hit.
3. **Production operations are under-covered.** Real self-hosted deployments also involve Kubernetes-based serving, autoscaling, observability and GPU cluster scheduling. The ~10 hr production ops block in Phase 2 addresses this; it's the cheapest gap to close given a data-engineering background.
4. **Phase 1 may feel deep.** It stays because you can't profile what you can't read, from Python down to CUDA. If behind schedule, cut the attention block and the last two matmul kernels, not the profiling.
5. **The tools churn fast.** Kernel DSLs, quantization formats and vLLM internals change monthly; roofline math, memory hierarchy and scheduling don't. Re-check the resource list every quarter; don't re-plan.
6. **Burnout is the real risk.** ~470 hours of evenings on top of a full-time job is a lot. One full rest week every 8 weeks is built into the slack.

### Deliberately left out

| Cut | Why |
|---|---|
| CuTe DSL / CUTLASS / TileLang / cuTile | Kernel-engineer depth. The focus here is reading and profiling kernels, not authoring GEMMs. A good follow-on once this plan is done |
| Distributed training | A different world from inference. Worth its own journey later |
| Certifications | Building things and contributing upstream teaches more |
| Buying a GPU | Prices are inflated and a used 3090 lacks FP8. Rent until usage passes ~15–20 hrs/week |
| AMD ROCm | Get exposure through GPU MODE competitions rather than a dedicated block |

**Trend to watch:** LLM agents writing kernels is a major 2026 research area. That lowers the value of raw kernel typing and raises the value of what this plan emphasizes: knowing the hardware limit, profiling, and verifying that fast kernels are also correct.

---

## Repo layout

Suggested structure as work lands (create folders when there's something to put in them):

```
gpu-inference-journey/
├── phase0-foundations/     # vLLM week-1 benchmark numbers, first CUDA kernels, roofline notes
├── phase1-kernels/         # matmul progression, Triton exercises, profiling reports
├── phase2-inference/       # vLLM request trace notes, quantization + parallelism experiments
├── deliverables/           # links to standalone deliverable repos / blog posts
├── notes/                  # weekly log (one entry per Sunday commit)
└── README.md
```

The Phase 1 kernel deliverable and Phase 2 Deliverable A should be **their own public repos** so they stand on their own; link them from `deliverables/`.

---

## Sources

- **GPU pricing:** Thunder Compute H100 tracker · Spheron RunPod vs Vast.ai comparison
- **Hour estimates** are rough, based on a typical pace for someone with strong SQL/Python and limited C++. Adjust after Phase 0 using actual pace.
