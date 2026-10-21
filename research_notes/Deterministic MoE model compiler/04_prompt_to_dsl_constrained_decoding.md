# Prompt-to-DSL / IR compilation, grammar-constrained decoding, and prompt-sensitivity mitigation (state as of 2026-10-21)

Research method note: about 45 tool calls (web search, arXiv/GitHub/HN fetches). The GitHub REST API was blocked in this session, so star counts come from the rendered GitHub pages read on 2026-10-21, are approximate, and were summarized by a fetch tool. The shared web-search budget ran out near the end, so a few follow-ups (GPTCache, Instructor/Pydantic AI/Marvin/Mirascope release status, recent "structured output hurts reasoning" studies) are listed under Gaps rather than guessed. "[vendor claim]" marks numbers reported by the project's own authors with no independent replication found.

## 1. Prompt -> DSL -> action: practice, frameworks, and "LLM as compiler front-end"

### Takeaway
"NL -> typed DSL/IR -> deterministic executor" is now a recognized design pattern. As of 2025-2026 it is backed by a Thoughtworks/martinfowler.com essay, research systems (LLMCompiler, Blueprint-First/Source Code Agent, Grammar Prompting) and declarative prompt languages (PDL, BAML, DSPy, TypeChat, LMQL, Pipelex). The evidence that it improves reliability is mostly qualitative or self-reported. The best quantitative support comes from planner/executor separation (TravelPlanner violations 275 -> 11) and DAG planning (up to 3.7x lower latency than ReAct). Free-form NL -> IR is still hard at enterprise scale (Spider 2.0: 21.3%).

### Cited Findings
**Practitioner discussion**
- (2026-07-14) Unmesh Joshi (Thoughtworks) on martinfowler.com, "DSLs Enable Reliable Use of LLMs". A DSL removes syntactic variation, and its deterministic validator (parser, schema, type checker or compiler) lets an agent generate, validate and repair with no human in the loop, using domain-level error messages. The generated DSL program, not the prompt, becomes the maintained source of truth. The approach works only for "small, well-factored, genuinely constrained DSLs with a validator", which carry real upfront design cost. The article reports no quantitative results — [martinfowler.com](https://martinfowler.com/articles/llm-and-dsls.html)
- (2025-05-18/19) HN thread on "Assemblers and compilers are (practically) deterministic. LLMs are not":
  - chowells argued the real distinction is defined semantics and a notion of correctness, so the goal should be predictability rather than determinism alone.
  - mmoskal: changing one letter in a prompt "will change the results in unpredictable ways".
  - vlovich123: seeding helps locally, but multi-tenant batching adds nondeterminism.
  - ModernMech: "programming languages are not for communicating with a computer, they are for communicating ideas in an unambiguous way."
  - [HN item 44020203](https://news.ycombinator.com/item?id=44020203)
- (2025-10-28) "Show HN: Pipelex – Declarative language for repeatable AI workflows" (122 points, 27 comments). The maintainer says workflows are meant to be "repeatable and deterministic, like a tool", are interpreted at runtime rather than compiled to Python, and can be tested pipe by pipe. A failed step currently restarts the whole run, and Temporal integration is planned. On BAML vs Pipelex: BAML = "typed single-call functions that generate SDKs", Pipelex = a multi-step workflow engine — [HN 45734865](https://news.ycombinator.com/item?id=45734865); [GitHub Pipelex](https://github.com/Pipelex/pipelex)
- (2026) A long tail of low-engagement Show HN projects follows the same "typed language instead of free-form agent" idea. Each has 1–12 points, which suggests an emerging pattern rather than a consolidated tool:
  - "Keel – Give your agent a typed strategy language, not a trading account" (2026-10-01): [HN 49924208](https://news.ycombinator.com/item?id=49924208)
  - "Constrained DSL for Reliable LLM Decisions" (2026-02-17): [HN 47042814](https://news.ycombinator.com/item?id=47042814)
  - "Topolog – Plans as typed DAG programs" (2026-06-10): [HN 48484353](https://news.ycombinator.com/item?id=48484353)
  - "Tagless final DSL for deterministic TypeScript programs" (2026-02-23): [HN 47119083](https://news.ycombinator.com/item?id=47119083)
  - "SLayer, a semantic layer maintained by your agent" (2026-05-11): [HN 48095686](https://news.ycombinator.com/item?id=48095686)
- (2024) SPML, a meta-language for prompt-attack defense, compiles system prompts into an IR to support analysis and transformation — [HN 39522245](https://news.ycombinator.com/item?id=39522245)
- (2026) CPPL is a domain example: a Python front-end DSL plus a JSON IR that the LLM generates, with deterministic lowering to CIRCT. The LLM emits structured IR instead of raw HDL. Seen via search snippet only — [arXiv 2605.17892](https://arxiv.org/html/2605.17892v1)

**Research systems (planner / IR / executor split)**
- (2025-08-01, v2 2026-06-16) "Blueprint First, Model Second" (Source Code Agent):
  - Design: an expert procedure is encoded as a source-code "Execution Blueprint" and run by a deterministic engine. The LLM is called only for bounded sub-tasks and never chooses the workflow path.
  - Results on TravelPlanner with Claude-Sonnet-4: final pass rate 35.56% vs 18.00% for ATLAS (+97.6% relative); constraint violations 11 vs 275 (−96.0%); 10.2 vs 14.0 steps.
  - [arXiv 2508.02721](https://arxiv.org/abs/2508.02721)
  - The same paper cites Logic-LM (NL -> symbolic form -> deterministic solver) as reporting a 39.2% gain over pure LLM reasoning (secondary citation) — [arXiv 2508.02721 PDF](https://arxiv.org/pdf/2508.02721)
- (ICML 2024) LLMCompiler: a Function Calling Planner emits a DAG of tasks with placeholder variables, a Task Fetching Unit dispatches ready tasks, and an Executor runs them in parallel. Compared with ReAct: up to 3.7x lower latency, up to 6.7x lower cost, up to about 9% higher accuracy. It is up to 1.35x faster than OpenAI parallel function calling at similar accuracy — [arXiv 2312.04511](https://arxiv.org/abs/2312.04511v2); [PMLR](https://proceedings.mlr.press/v235/kim24y.html); code [SqueezeAILab/LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler)
- (NeurIPS 2023) Grammar Prompting (Wang et al., Google/MIT):
  - Each demonstration carries a minimal BNF subset of the DSL grammar. At test time the model first predicts a specialized BNF grammar, then generates output under it.
  - Prepending the full grammar "did not yield any improvements".
  - Tasks: SMCalFlow, Overnight, GeoQuery semantic parsing, PDDL planning, SMILES.
  - [arXiv 2305.19234](https://arxiv.org/pdf/2305.19234); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd40d0d65bfebb894ccc9ea822b47fa8-Abstract-Conference.html)
- (ICLR 2025 oral) Spider 2.0 measures how hard NL -> SQL is at enterprise scale. An o1-preview code-agent solves 21.3% of 632 tasks, vs 91.2% on Spider 1.0 and 73.0% on BIRD — [arXiv 2411.07763](https://arxiv.org/abs/2411.07763)
- (NeurIPS 2024) SGLang's frontend language and runtime treat LM programs as structured programs and report up to 6.4x higher throughput across workloads — [arXiv 2312.07104](https://arxiv.org/html/2312.07104v2)

**Declarative prompt languages / typed-LLM-function frameworks (snapshot 2026-10-21)**
- **IBM PDL** — "a simple declarative data-oriented language that puts prompts at the forefront, based on YAML" (arXiv 2024-10-24) — [arXiv 2410.19135](https://arxiv.org/abs/2410.19135)
  - Repo: about 312 stars. Features: type checking of model input/output, control flow, tools, trace telemetry, a live-document GUI. Follow-on papers: AutoPDL (AutoML'25), PPDL (ICML'26).
  - The repo page says it "was archived by the owner on Oct 21, 2026" and is read-only — [GitHub IBM/prompt-declaration-language](https://github.com/IBM/prompt-declaration-language)
- **BAML** (BoundaryML) — about 9.4k stars. TypeScript-like syntax, Rust-like type system, "types persist at runtime", functions callable from Python, TS, Go, C# and Java — [GitHub BoundaryML/baml](https://github.com/BoundaryML/baml)
  - Schema-Aligned Parsing (SAP): lenient, schema-guided parsing (strips "yapping", fixes commas and quotes, casts types, completes partial streams, picks the best candidate) instead of constrained decoding — [BAML blog](https://www.boundaryml.com/blog/schema-aligned-parsing)
  - [vendor claim, Aug 2024] Function-calling accuracy on BFCL, n=1000 per model:

    | Model | Native FC | SAP |
    |---|---|---|
    | gpt-4o-mini | 19.8% | 92.4% |
    | claude-3-haiku | 57.3% | 91.7% |
    | gpt-4o | 87.4% | 93% |

    Sources: [BAML blog](https://www.boundaryml.com/blog/schema-aligned-parsing); [BoundaryML SOTA FC post](https://boundaryml.com/blog/sota-function-calling)
- **DSPy** — about 38.7k stars, "programming, rather than prompting". Signatures are typed I/O specs and optimizers "compile" programs — [GitHub stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
  - GEPA reflective prompt optimizer (ICLR 2026 oral): beats GRPO by 6% on average (up to 20%) with up to 35x fewer rollouts, and beats MIPROv2 by more than 10% (+12% on AIME-2025) — [arXiv 2507.19457](https://arxiv.org/abs/2507.19457)
  - DSPy 3.3.0 adds experimental `dspy.Flex`, which lets GEPA rewrite a module's implementation, not just its instructions (secondary sources) — [newreleases.io](https://newreleases.io/project/github/stanfordnlp/dspy/release/3.3.0); [magica.com](https://magica.com/news/dspy-flex-optimizable-module-code)
- **TypeChat** (Microsoft) — about 8.7k stars. Replaces "prompt engineering with schema engineering":
  - Intents are a TypeScript discriminated union, and a "meta-schema" routes to sub-schemas.
  - Responses are validated against the types, and failures are sent back to the model with validator errors for repair.
  - The result is summarized without an LLM so the user can confirm intent.
  - [GitHub microsoft/TypeChat](https://github.com/microsoft/TypeChat); [TypeChat 0.1.0 blog](https://microsoft.github.io/TypeChat/blog/announcing-typechat-0-1-0/)
- **LMQL** (ETH SRI) — about 4.2k stars. A Python superset with `where` constraints, types and stop phrases, plus decoders (argmax, beam, best_k) — [GitHub eth-sri/lmql](https://github.com/eth-sri/lmql)
- **Outlines** (.txt) — about 16.0k stars. Regex, JSON/Pydantic, choice and CFG output types. Backends: vLLM, Ollama, transformers, llama.cpp, OpenAI, Gemini, Dottxt — [GitHub dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)

### Inferences
- The industry consensus (Fowler, TypeChat, Blueprint-First, Pipelex) puts the LLM only at the NL -> IR boundary, or at bounded leaf tasks, and makes everything downstream of the IR deterministic code. That is the "compiler front-end" framing. Determinism of the whole system then reduces to determinism of the NL -> IR step plus a pure executor.
- Grammar Prompting's result — a per-example minimal grammar helps, a full grammar dump does not — suggests showing the parser model only the relevant slice of a large IR schema, e.g. through a two-stage intent -> sub-schema route as in TypeChat's meta-schema.
- Spider 2.0's 21.3% is a warning. When the IR must reference large, messy external state (enterprise schemas), NL -> IR accuracy, not determinism, is the bottleneck. Keep the IR's vocabulary small and closed.
- PDL being archived on 2026-10-21 suggests IBM is consolidating. Do not build a dependency on PDL.

### Gaps
- Could not verify current stars or releases for Instructor, Pydantic AI, Marvin, Mirascope, Guidance (python) or SGLang (GitHub API blocked; search budget exhausted).
- Found no independent replication of BAML's SAP BFCL numbers.
- Found no large, rigorous 2026 study comparing NL -> DSL -> executor against direct agentic execution on determinism metrics (e.g. pass^k, output-hash agreement). Evidence is mostly single papers and qualitative essays.
- No primary source found for a named "deterministic agent" framework with production adoption beyond those listed. The X/Twitter and Reddit r/LocalLLaMA discussions could not be searched.

## 2. Grammar/structure-constrained decoding engines: guarantees, overheads, accuracy effects

### Takeaway
As of October 2026 the open-source field has consolidated on two engines, XGrammar (now XGrammar-2) and llguidance. Both use Earley-style CFG parsing with token-mask caches. Both report near-zero end-to-end overhead: llguidance about 50 µs CPU per token, XGrammar-2 about 6x faster compilation than prior engines and up to 80x as tool counts grow. They are integrated into vLLM, SGLang, TensorRT-LLM and llama.cpp. Closed APIs (OpenAI, Anthropic) guarantee schema conformance only for a conservative subset of JSON Schema, with documented exceptions (refusals, max_tokens, Anthropic enum casing).

On accuracy, the 2024 "Let Me Speak Freely" claim that format constraints hurt reasoning was largely rebutted for well-prompted, well-implemented constrained decoding (+1 to +4 points on GSM8K and similar tasks). Real risks remain in three places: grammars too narrow to leave room for reasoning (CRANE), token-boundary misalignment (DOMINO), and distribution distortion (GAD).

### Cited Findings
**Engines**

*XGrammar (v1)* (MLSys 2025; arXiv 2024-11-22)
- Design: splits the vocabulary into context-independent tokens (precomputed) and context-dependent tokens (checked at runtime), uses a persistent stack, and overlaps grammar work with GPU compute.
- Results: "up to 100x speedup over existing solutions" and "near-zero" end-to-end overhead.
- [arXiv 2411.15100](https://arxiv.org/abs/2411.15100)
- Integration timeline: SGLang and MLC-LLM (2024/11), vLLM (2024/12), TensorRT-LLM (2025/01), Modular MAX (2025/02), OpenVINO GenAI (2025/09). The README claims it is "the default structured generation backend for most LLM inference engines" [vendor claim]. About 2.0k stars — [GitHub mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)

*XGrammar-2* (arXiv v1 2026-01-07, v4 2026-08-05; ACM CAIS '26)
- Design: built for dynamic agentic structures (tool calls, response protocols). TagDispatch switches grammar when a tag is triggered. A Cross-Grammar Cache reuses substructures across grammars. Earley-based adaptive token-mask cache, JIT compilation and repetition-state compression.
- Results: "over 6x faster compilation than prior structured generation engines"; near-zero end-to-end overhead.
- [arXiv 2601.04426](https://arxiv.org/abs/2601.04426)
- (2026-05-04 MLC blog) [vendor claims]:
  - Up to 80x compile speedup over XGrammar v1 as tool count scales from 10 to 500.
  - Repetition compression cuts preprocessing from 534 ms to 5.37 ms (O(count) -> O(1)).
  - About 50% of structures are reused across 50 tool schemas.
  - 100% schema accuracy with Structural Tag on BFCL-V3 simple and parallel subsets for four models.
  - Supports the OpenAI Harmony format, reasoning channels, and speculative decoding via `traverse_draft_tree`.
  - Claimed adopters: xAI, Databricks, DeepSeek. Engines: SGLang, vLLM, TRT-LLM, MLC-LLM.
  - [MLC blog](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation)

*llguidance* (Microsoft/guidance-ai; v1.0.0 declared 2025-06-23; about 892 stars)
- Design: Earley CFG parser plus a lexer built on regex derivatives, with token-trie traversal for masks.
- Performance [vendor claim]: about 50 µs CPU per token for a 128k tokenizer. A full mask for a typical JSON schema takes about 1.5 ms, but thanks to the "slicer" optimization the average on JSONSchemaBench is under 50 µs. Fewer than 1% of masks take over 1 ms, and 0.001% take over 10 ms. The README claims batch sizes up to 3200 with 16 cores and a 10 ms forward pass.
- Integrations: powers OpenAI Structured Outputs for JSON Schema (shipped 2025-05-20); vLLM v0.8.2 (2025-03-25); SGLang v0.4.4 (2025-02-26, `--grammar-backend llguidance`); llama.cpp b4613 (2025-02-01); Chromium `window.ai` (2025-04-11).
- [GitHub guidance-ai/llguidance](https://github.com/guidance-ai/llguidance)

*Outlines* (Willard & Louf, arXiv 2023-07-19)
- Design: recasts generation as transitions in a regex/CFG-derived FSM, with a precomputed vocabulary index.
- Result: guaranteed structure with "little overhead".
- [arXiv 2307.09702](https://arxiv.org/abs/2307.09702)

*SGLang compressed FSM* (2024-02-05)
- Design: compresses singular-transition paths in the FSM so multiple tokens are decoded in one step.
- Result: up to 2x lower latency and 2.5x higher throughput than guidance+llama.cpp and outlines+vLLM.
- [LMSYS blog](https://lmsys.org/blog/2024-02-05-compressed-fsm)

*vLLM structured outputs* (docs, current)
- Backends: `xgrammar` and `guidance` (llguidance), plus `auto` per-request selection.
- Constraint types: `choice`, `regex`, `json`, `grammar` (EBNF CFG) and `structural_tag`.
- Legacy `guided_*` parameters were removed in v0.12.0.
- Reasoning models: needs `--reasoning-parser`. Qwen3-Coder requires `enable_in_reasoning=True`. Grammars that include `</think>` need token-level elements in XGrammar.
- The docs' tip: stating the schema in the prompt "can improve the results notably".
- [vLLM docs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)

**Closed-API guarantees**

*OpenAI Structured Outputs* (2024-08-06)
- `strict: true` on function definitions, or `response_format: json_schema`.
- gpt-4o-2024-08-06 scores 100% on OpenAI's internal complex-schema eval, vs under 40% for gpt-4-0613.
- Exceptions: refusals and max-token truncation.
- [OpenAI blog](https://openai.com/index/introducing-structured-outputs-in-the-api/) (page returned 403 to the fetcher; figures via search snippet and [Gigazine](https://gigazine.net/gsc_news/en/20240807-openai-structured-outputs))
- A community thread claims strict mode sometimes fails to enforce the schema (anecdotal) — [OpenAI forum](https://community.openai.com/t/strict-mode-does-not-enforce-the-json-schema/1104630)

*Anthropic Structured Outputs* (docs, current) — all from [Claude docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- Two features: `output_config.format` (JSON schema) and `strict: true` tool use. Both compile the schema into a grammar for constrained decoding. GA on Claude API, Bedrock, Google Cloud and Foundry.
- Supported schema subset: no recursive schemas, numeric min/max, string length limits, `minItems`/`maxItems` beyond 0 or 1, or lookaround/backreference regex. `additionalProperties: false` is required.
- Complexity limits per request: 20 strict tools, 24 optional parameters, 16 union-typed parameters. There are also internal grammar-size limits ("Schema is too complex for compilation") and a 180 s compile timeout.
- Grammar caching: the first request with a new schema pays compile latency. Grammars are cached 24 h from last use, keyed on schema structure and tool set; tool name and description changes do not invalidate.
- Caveats:
  - On `stop_reason: refusal` or `max_tokens`, output may not match the schema.
  - Enum/const values "may differ from the schema only in capitalization", so the docs advise case-insensitive comparison.
  - Output property order is required properties first (in schema order), then optional ones.
  - Changing the format invalidates the prompt cache, and an injected system prompt adds input tokens.

**Coverage / overhead benchmark**

*JSONSchemaBench* (Geng et al., Microsoft Research et al.; arXiv 2501.10868; ICML 2025 listing): about 10K real-world schemas. [arXiv HTML](https://arxiv.org/html/2501.10868v2); [ICML](https://icml.cc/virtual/2025/51849)

- Empirical coverage, GitHub-Hard subset:

  | Engine | Coverage |
  |---|---|
  | Guidance (llguidance) | 0.41 |
  | llama.cpp | 0.39 |
  | XGrammar | 0.28 |
  | OpenAI | 0.09 |
  | Outlines | 0.03 |
  | LM-only | 0.13 |

- Other subsets:
  - Kubernetes: Guidance 0.91 vs XGrammar 0.07.
  - GlaiveAI (easy): every engine is at least 0.86.
  - OpenAI/Gemini coverage is low because they support only a conservative feature subset, but compliance is about 1.00 where supported.
- Efficiency, llama.cpp backend, medians:
  - Guidance: grammar compile time (GCT) 0.00–0.01 s; time per output token (TPOT) 6.4–9.5 ms, vs about 15–17 ms LM-only, because of fast-forwarding.
  - Outlines: GCT 3.48–8.05 s; TPOT 30–47 ms. Features like `minItems`/`maxItems`/`enum` can take 40 s to 10 min to compile, which explains Outlines' timeouts.
  - Transformers backend: XGrammar GCT 0.12–0.30 s (hidden behind prefill), TPOT about 65 ms vs Guidance about 36 ms.
- The paper says constrained decoding "can speed up the generation process by 50%".
- Correctness: on the JSON Schema Test Suite, Guidance had the fewest failures. XGrammar had the most under-constrained failures (38 categories).

**Accuracy effects**

*"Let Me Speak Freely?"* (Tam et al.; arXiv v1 2024-08-05, v3 2024-10-14; EMNLP 2024 Industry) — reports a "significant decline" in reasoning under format restrictions (JSON-mode, stricter constraints = larger drops), but better classification accuracy — [arXiv 2408.02442](https://arxiv.org/abs/2408.02442)

*.txt rebuttal* ("Say What You Mean", Will Kurt) — [dottxt blog](https://blog.dottxt.ai/say-what-you-mean.html)
- Objections: the original's NL and JSON prompts differ; the JSON prompts never specify a schema; an LLM "perfect parser" (claude-3-haiku) underperforms a flexible regex (0.57 vs 0.61); and JSON-mode is conflated with true constrained decoding.
- With Llama-3-8B-Instruct, structured generation equals or beats unstructured:

  | Task | Unstructured | Structured |
  |---|---|---|
  | GSM8K | 0.77 | 0.78 |
  | Last Letter | 0.73 | 0.77 |
  | Shuffle Objects | 0.41 | 0.44 |

*JSONSchemaBench quality* (Table 8) — all constrained engines are at or above the LM-only baseline; Guidance's edge is attributed to token healing — [arXiv HTML](https://arxiv.org/html/2501.10868v2)

| Task | LM-only | XGrammar | Guidance |
|---|---|---|---|
| GSM8K | 80.1% | 83.7% | 83.8% |
| Last Letters | 50.7% | 51.2% | 54.0% |
| Shuffle Objects | 52.6% | 52.7% | 55.9% |

*CRANE* (ICML 2025) — gives a theoretical argument that grammars allowing only the final answer reduce reasoning. Augmenting the grammar so a free reasoning region precedes the constrained answer preserves reasoning: up to +10 percentage points on GSM-Symbolic and FOLIO — [PMLR](https://proceedings.mlr.press/v267/banerjee25a.html); [arXiv 2502.09061](https://arxiv.org/abs/2502.09061v3)

*DOMINO* (ICML 2024) — constrained decoding that is misaligned with subword tokenization hurts accuracy. Subword-aligned "minimally invasive" decoding with precomputation and speculation has "virtually no overhead" and in some cases almost 2x speedup — [PMLR](https://proceedings.mlr.press/v235/beurer-kellner24a.html); [arXiv 2403.06988](https://arxiv.org/pdf/2403.06988)

*Grammar-Aligned Decoding / ASAp* (NeurIPS 2024) — standard grammar-constrained decoding distorts the LLM's distribution (grammatical but improbable outputs). ASAp converges toward the grammar-conditioned distribution using past samples — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2bdc2267c3d7d01523e2e17ac0a754f3-Abstract.html). A secondary review notes little downstream effect on SyGuS and slow convergence in some cases — [Moonlight review](https://www.themoonlight.io/review/grammar-aligned-decoding)

*Type-constrained decoding for code* (Mündler et al., PLDI 2025) — prefix automata plus a search over inhabitable types enforce well-typedness (TypeScript). This cuts compilation errors by more than half on HumanEval/MBPP and improves functional correctness for synthesis, translation and repair, including on models over 30B — [arXiv 2504.09246](https://arxiv.org/abs/2504.09246); [PLDI](https://pldi25.sigplan.org/details/pldi-2025-papers/25/Type-Aware-Constraining-for-LLM-Code-Generation)

### Inferences
- Engine choice for the NL -> IR step:
  - llguidance has the broadest JSON-Schema coverage and correctness in JSONSchemaBench and the lowest per-token cost.
  - XGrammar-2 is strongest for many-tool, dynamic-protocol (structural tag) workloads and is the most widely integrated.
  - Both are available in vLLM and SGLang, so a deterministic compiler can pin one and treat the other as a cross-check.
  - Outlines' compile-time blowups make it unsuitable for large or complex IR schemas.
- The engine guarantees syntactic conformance only. Semantic validity (cross-field constraints, referential integrity) still needs a post-decode validator. Hosted APIs reject numeric and length constraints entirely, so those must move into validators.
- Practical pattern for accuracy:
  - Let the model reason in an unconstrained (or loosely constrained) region, then constrain the IR emission (CRANE; vLLM reasoning parser).
  - Put the schema in the prompt as well (vLLM tip; .txt rebuttal).
  - Order keys so that "reasoning/evidence" fields come before decision fields. Speak Freely reported that key order interferes with reasoning, and Anthropic reorders required-before-optional, so make all fields required to control order.
- Anthropic's documented enum-casing deviation means a canonicalizer must case-normalize enums even under strict mode, or the IR should avoid enum values that differ only by case.

### Gaps
- No independent benchmark of XGrammar-2 vs llguidance on the same hardware and schemas in 2026. JSONSchemaBench predates XGrammar-2 and llguidance 1.0.
- OpenAI's exact current implementation (beyond llguidance powering JSON-Schema structured outputs per the llguidance README) and current schema-subset limits were not re-verified (the blog returned 403).
- Search budget ran out before locating 2025–2026 follow-up studies on structured output vs reasoning beyond CRANE and the .txt rebuttal.
- No per-token overhead figure for XGrammar-2 was published on the pages read; it is only described as "near-zero".

## 3. Prompt sensitivity: quantification and mitigation (canonicalization, ensembling, consistency training)

### Takeaway
Meaning-preserving format and paraphrase changes swing accuracy by up to 76 points (LLaMA-2-13B, FormatSpread). Scale and instruction tuning do not reliably remove this (FormatSpread, POSIX), though larger models are somewhat more robust (ProSA). Few-shot exemplars are the cheapest reliable reducer (POSIX, ProSA). Training-time fixes exist: Mixture of Formats, Flip-Flop Consistency (+11.6% agreement), latent adversarial paraphrasing. Even a perfectly canonical prompt does not give identical outputs unless inference is batch-invariant: 80 unique completions out of 1000 at temperature 0 on standard vLLM.

### Cited Findings
- **FormatSpread** (Sclar et al.; arXiv 2023-10-17; ICLR 2024)
  - Up to 76 accuracy points of spread from formatting alone (LLaMA-2-13B, few-shot).
  - Sensitivity persists with model size, more shots and instruction tuning.
  - Format rankings correlate weakly across models.
  - FormatSpread estimates the performance interval over plausible formats without access to weights.
  - [arXiv 2310.11324](https://arxiv.org/abs/2310.11324)
- **POSIX** (EMNLP 2024 Findings; arXiv 2024-10-03)
  - Metric: the relative change in a response's log-likelihood under intent-preserving prompt swaps.
  - Larger models or instruction tuning do "not necessarily" reduce sensitivity.
  - Even one few-shot exemplar "almost always leads to significant decrease".
  - Template changes are worst for MCQ tasks; paraphrases are worst for open-ended generation.
  - [arXiv 2410.02185](https://arxiv.org/abs/2410.02185)
- **ProSA** (EMNLP 2024 Findings; arXiv 2024-10-16)
  - Metric: instance-level PromptSensiScore.
  - Larger models are more robust, few-shot reduces sensitivity, and higher decoding confidence correlates with robustness.
  - Subjective evaluations are also affected.
  - [arXiv 2410.12405](https://arxiv.org/abs/2410.12405)
- **State of What Art?** (Mizrahi et al.; TACL; arXiv v3 2024-05-06) — 6.5M instances, 20 LLMs, 39 tasks, 3 benchmarks. Single-template evaluation is brittle, so the authors call for multi-prompt evaluation with metrics tailored to the use case — [arXiv 2401.00595](https://arxiv.org/abs/2401.00595)
- **Mixture of Formats (MOF)** (Ngweta et al., RPI/IBM; NAACL 2025 SRW; arXiv 2025-04-09) — diversifying the style of few-shot exemplars reduces style-induced brittleness and improves performance across prompt variations — [arXiv 2504.06969](https://arxiv.org/abs/2504.06969v1); [ACL Anthology](https://aclanthology.org/2025.naacl-srw.51)
- **Flip-Flop Consistency (F²C)** (ACL 2026; arXiv v1 2025-10-16, v2 2026-10-05)
  - Method: unsupervised training with Consensus Cross-Entropy (majority-vote pseudo-label across prompt variants) plus representation alignment toward the high-confidence majority.
  - Evaluation: 11 datasets, 4 tasks, 4–15 variants each.
  - Results: observed agreement +11.62%, mean F1 +8.94%, cross-format variance −3.29%. It generalizes to held-out formats.
  - [arXiv 2510.14242](https://arxiv.org/abs/2510.14242)
  - F²C's related work (secondary): a supervised contrastive paraphrase approach (Yan et al. 2024) gained about 2.5% on unseen perturbations. It also argues that prompt-level fixes (paraphrase generators, answer selection) do not fix internal inconsistency — [arXiv 2510.14242 PDF](https://arxiv.org/pdf/2510.14242)
- **Latent Adversarial Paraphrasing (LAP)** (EMNLP 2025) — trains against a learned continuous "latent paraphrase" perturbation, improving worst-case and average win rates with no extra paraphrase data or inference latency (search-snippet level) — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1595.pdf)
- **Related 2025–2026 lines** (titles seen in search; not read in full):
  - Prompt-Reverse Inconsistency: [arXiv 2504.01282](https://arxiv.org/abs/2504.01282)
  - JudgeSense, prompt sensitivity of LLM-as-judge: [arXiv 2604.23478](https://arxiv.org/html/2604.23478v1)
  - "Position: It's Time to Optimize LLMs for Self-Consistency": [arXiv 2608.05188](https://arxiv.org/pdf/2608.05188)
  - Bias-Augmented Consistency Training: [arXiv 2403.05518](https://arxiv.org/pdf/2403.05518)
- **Inference-level nondeterminism** (Horace He / Thinking Machines, 2025-09-10)
  - Experiment: Qwen3-235B-A22B-Instruct (an MoE model) at temperature 0, same prompt, 1000 runs.
  - Standard vLLM gave 80 unique completions, identical for 102 tokens and diverging at token 103. Batch-invariant kernels (RMSNorm, matmul, attention) gave 1 unique completion.
  - Root cause: lack of batch invariance under varying server load, not atomics.
  - Cost on Qwen3-8B: 26 s default vs 55 s unoptimized deterministic vs 42 s with the improved attention kernel.
  - [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/); [Simon Willison commentary](https://simonwillison.net/2025/Sep/11/defeating-nondeterminism)
- **DSPy/GEPA** — prompt optimization treats the prompt as a compiled artifact rather than hand-written text, which shrinks human-induced prompt variance — [arXiv 2507.19457](https://arxiv.org/abs/2507.19457)

### Inferences
- For a "deterministic compiler", sensitivity has two separable sources, each needing its own fix:
  - Semantic: different NL leads to a different IR. Fixes: canonical IR, few-shot anchors, consistency-trained parser, clarification.
  - Numeric: the same tokens give different outputs. Fixes: batch-invariant kernels, fixed decoding config, record/replay.
- The Thinking Machines experiment used an MoE model, so MoE serving is subject to the same batch-dependence. Expert routing, a top-k over logits, plausibly amplifies tiny numeric differences into different expert choices. This is an inference; the blog did not isolate routing.
- Normalizing the user prompt (lowercasing, whitespace, template stripping) only addresses format-level variance (FormatSpread/POSIX template effects). Paraphrase-level variance, the main driver for open-ended generation per POSIX, has to be absorbed by mapping to a closed IR vocabulary. That makes the NL -> IR parser the component to harden: few-shot plus MOF-style diverse exemplars plus F²C-style consistency fine-tuning on paraphrase sets.
- Paraphrase ensembling (parse k paraphrases, majority-vote the canonical IR) is the prompt-level analogue of F²C's consensus label. It trades k-times cost for agreement and naturally produces a disagreement signal for clarification. No source measured this exact recipe.

### Gaps
- No 2025–2026 source found that measures an end-to-end "prompt canonicalization -> IR" pipeline's paraphrase-invariance rate (e.g. % of paraphrase sets mapping to an identical IR hash).
- POSIX/ProSA numeric tables were not retrieved (abstract-level only).
- No published numbers found for frontier closed models' format sensitivity in 2026.
- LAP's exact numbers were not verified.

## 4. Designing the IR: typed schemas, intent taxonomies, slots, ambiguity/clarification, versioning, round-trip checks, achievable determinism

### Takeaway
Published guidance converges on these principles:
- Make the IR a closed, typed schema: a discriminated union of intents with typed slots.
- Keep it small and fit to hosted-API grammar limits (Anthropic: 24 optional and 16 union params per request).
- Validate deterministically and repair from validator errors (TypeChat, Fowler).
- Show a non-LLM rendering of the IR for confirmation (TypeChat).
- Detect ambiguity explicitly and ask, rather than guess (AMBROSIA/PRACTIQ show models are poor at this unaided).
- Use round-trip (IR -> NL -> IR) agreement as an unsupervised correctness signal (RTC, ICML 2024).

Bitwise determinism of the NL -> IR step is achievable in principle by combining greedy decoding, constrained decoding, batch-invariant kernels, canonical serialization and an exact-match cache. Semantic invariance across paraphrases is not guaranteed and must be measured.

### Cited Findings
- **TypeChat schema design**
  - Intents are members of a discriminated union, and a meta-schema routes to sub-schemas for hierarchical apps.
  - Validation errors drive a repair turn.
  - A summary of the instance is produced without an LLM so the user can confirm intent.
  - [GitHub microsoft/TypeChat](https://github.com/microsoft/TypeChat); [TypeChat 0.1.0 blog](https://microsoft.github.io/TypeChat/blog/announcing-typechat-0-1-0/)
- **DSL design rules** (2026-07-14): small, well-factored, constrained DSLs with domain-level validator errors. Progressive interfaces make invalid programs (e.g. a step before topology) fail to compile at the exact illegal step, so the LLM can self-correct — [martinfowler.com](https://martinfowler.com/articles/llm-and-dsls.html)
- **Hosted grammar limits shape IR design** (Anthropic): 20 strict tools, 24 optional params and 16 union-typed params per request; no recursion; no numeric or length bounds; a 180 s compile timeout. Recommended mitigations: make parameters required, flatten nesting, split tools — [Claude docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- **Canonical ordering**: Anthropic emits required properties first, then optional, each in schema order — [Claude docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs). Key order can interfere with reasoning — [arXiv 2408.02442](https://arxiv.org/abs/2408.02442)
- **Reasoning slot before decision slots** preserves accuracy under constraints — [CRANE, PMLR](https://proceedings.mlr.press/v267/banerjee25a.html)
- **Ambiguity**
  - AMBROSIA (NeurIPS 2024 D&B; Saparina & Lapata): text-to-SQL with scope ambiguity, attachment ambiguity and vagueness, where the ambiguity persists given the DB. "Even the most advanced models struggle to identify and interpret ambiguity" — [arXiv 2406.19073](https://arxiv.org/pdf/2406.19073); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a4c942a8405cc910f0a833d28d2573cc-Abstract.html)
  - PRACTIQ (AWS/UMass): conversational text-to-SQL with ambiguous and unanswerable questions, structured as clarifying question, user answer, then clarified SQL — [arXiv 2410.11076](https://arxiv.org/pdf/2410.11076)
- **Round-trip correctness (RTC)** (Allamanis, Panthaplackel, Yin; ICML 2024)
  - Method: predict an intermediate (e.g. an NL description of code), feed it back to regenerate the original form, and check semantic equivalence.
  - RTC correlates strongly with benchmark performance and needs no labels.
  - [arXiv 2402.08699](https://arxiv.org/abs/2402.08699v2); [PMLR](https://proceedings.mlr.press/v235/allamanis24a.html)
- **Typed IR plus a typed constraint engine** cuts invalid outputs by more than half for TypeScript — [arXiv 2504.09246](https://arxiv.org/abs/2504.09246)
- **DAG-structured plan IR** with placeholder variables enables parallel deterministic execution — [arXiv 2312.04511](https://arxiv.org/abs/2312.04511v2)
- **Code-as-blueprint IR** with an LLM only at bounded leaves cut constraint violations 96% — [arXiv 2508.02721](https://arxiv.org/abs/2508.02721)
- **Determinism of decoding**: temperature 0 alone is insufficient on shared servers. Batch-invariant kernels are required for bitwise-identical outputs, at roughly 1.6–2.1x runtime in the unoptimized vLLM prototype — [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **Tool-call / protocol IR**: XGrammar-2 Structural Tags express tool calling, Harmony format and reasoning channels as one composable JSON protocol — [MLC blog](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation). vLLM exposes `structural_tag` — [vLLM docs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)

### Inferences
Recommended NL -> IR stack, synthesized from the above. This is not a published recipe.
1. **Intent routing**: a closed intent enum, decoded constrained, which selects a per-intent sub-schema. This follows TypeChat's meta-schema and Grammar Prompting's minimal-grammar results.
2. **Slot filling**: a slot schema with all fields required and nullable instead of optional. This fixes order, avoids optional-param limits, and leaves the parser no choice of field order. Enums are closed and case-distinct, and units and dates are normalized in the IR.
3. **Explicit uncertainty**: an `ambiguities: [{slot, candidates}]` / `needs_clarification` field. The executor refuses to run when it is non-empty: ask, don't guess.
4. **Decoding**: greedy, constrained by llguidance or XGrammar, on a pinned model, tokenizer and engine version with batch-invariant kernels (or an isolated batch=1 lane). An optional unconstrained reasoning prefix precedes the IR (CRANE), discarded after parsing.
5. **Canonical serialization**: sorted keys, normalized numbers and strings, schema-version tag. Hash the result to get the cache key and identity.
6. **Validation**: semantic validators (cross-field, referential, numeric bounds the grammar engines can't express), with a bounded repair loop driven by validator error messages.
7. **Round trip**: render IR -> NL deterministically (template, no LLM, as in TypeChat), re-parse NL -> IR, and require hash equality. Any mismatch is treated as ambiguity, so ask for clarification. Optionally also parse k paraphrases and require agreement.
8. **Versioning**: a semantic version on the IR schema plus migration functions. The cache key includes (schema version, parser model hash, engine/grammar version).

Expected determinism:
- Bitwise determinism for a fixed input string is achievable given steps 4–5 plus record/replay.
- Paraphrase invariance is probabilistic and should be reported as a measured agreement rate, in the F²C/FormatSpread style, rather than claimed as guaranteed.

### Gaps
- Found no published benchmark of "IR -> NL -> IR" round-trip as an ambiguity detector for NL-to-DSL. RTC is for code.
- Found no source quantifying clarification-question policies (when to ask vs guess) in production DSL compilers.
- No primary source on IR schema-versioning practices specific to LLM front-ends; the recommendation above is inferred from general compiler practice.
- Small specialized parser models (fine-tuned semantic parsers vs frontier models) were not benchmarked in sources retrieved in this pass.

## 5. Semantic caching, memoization and record/replay for determinism

### Takeaway
Exact-match memoization keyed by a canonical IR hash is deterministic and safe. Embedding-similarity semantic caches are not: static cosine thresholds give no correctness guarantee, and correct and incorrect hits have overlapping similarity distributions. vCache (ICLR 2026) adds per-entry learned thresholds with a user-set error bound (up to 12.5x more hits, 26x lower error than baselines). Plan-level caching for agents (APC, NeurIPS 2025) cuts cost about 50%. Record/replay ("cassette") libraries for LLM calls are proliferating but young.

### Cited Findings
- **vCache** (Schroeder et al.; ICLR 2026; arXiv 2502.03771)
  - The authors call it "the first verified semantic cache with user-defined error rate guarantees". An online-learned per-cached-prompt threshold maximizes hit rate subject to the error bound, with no upfront training, and works with any embedding model.
  - Static thresholds give no formal guarantees and can produce unexpected error rates, because correct and incorrect hits overlap in similarity.
  - Results: up to 12.5x higher hit rate and 26x lower error rate vs static-threshold and fine-tuned-embedding baselines.
  - [arXiv 2502.03771](https://arxiv.org/html/2502.03771v5); [ICLR](https://iclr.cc/virtual/2026/poster/10006471)
- **"Similarity Is Not Validity"** (arXiv 2609.35908, 2026) — semantic caches are poisonable. Embeddings discard validity information, which limits embedding-only defenses. It replays vCache benchmarks at cosine 0.90 to measure false positives — [arXiv 2609.35908](https://arxiv.org/html/2609.35908v1)
- **"Closing the Operational Gap in Semantic Caching"** (EMNLP 2026 Industry; arXiv 2606.19719, v4 2026-09-01)
  - PR-AUC, the standard semantic-cache metric, ignores fixed-threshold behavior, and the highest-PR-AUC models are sometimes worst in deployment.
  - The authors propose P-CHR AUC and an Operational Retention Rate. The offline-to-deployed gap is partly structural (set by the positive rate) and fixable only by score re-normalization over the candidate pool or a changed training objective.
  - Title conflict: search results listed it as "Closing the Calibration Gap…", while the arXiv abs page shows "Operational Gap", probably a retitle across versions.
  - [arXiv 2606.19719](https://arxiv.org/abs/2606.19719)
- **Practitioner thresholds** [unverified blog figures]: typical production cosine thresholds 0.92–0.97, with false positives rising below 0.90; one blog reports 8–15% false positives at 0.90 on broad chat workloads — [llm-academy.dev](https://llm-academy.dev/gateway/semantic-caching/); [FutureAGI](https://futureagi.com/blog/what-is-semantic-caching-llms-2026/)
- **Agentic Plan Caching (APC)** (Zhang, Wornow, Wan, Olukotun; NeurIPS 2025; v2 2026-01-26)
  - Extracts structured plan templates from completed runs, matches new requests by keyword extraction, and adapts templates with lightweight models.
  - Results: 50.31% average cost reduction and 27.28% latency reduction with performance maintained.
  - The authors argue chatbot-style semantic caches fail for agents whose outputs depend on external state.
  - [arXiv 2506.14852](https://arxiv.org/abs/2506.14852)
- **Provider-side grammar caching** is keyed on schema structure: Anthropic caches compiled grammars 24 h from last use, invalidated by schema or tool-set changes but not by name or description edits — [Claude docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- **Record/replay libraries** (2025–2026, all young, adoption unknown):
  - standin (HTTP-level, LLM-aware matching, streaming, secret redaction): [PyPI](https://pypi.org/project/standin/)
  - cassette-fn (function-boundary recording, so retry and control flow still execute on replay): [PyPI](https://pypi.org/project/cassette-fn/)
  - cendor-cassette (whole agent run: every LLM and tool call in order): [PyPI](https://pypi.org/project/cendor-cassette/1.0.1/)
  - evalcraft (cassettes to pytest; replay covers plumbing, not quality): [PyPI](https://pypi.org/project/evalcraft/0.5.0/)
  - coding-agent-vcr (Claude Code/Codex proxy, early alpha): [PyPI](https://pypi.org/project/coding-agent-vcr/)
  - ReqCassette (Elixir, ReqLLM integration): [HexDocs](https://hexdocs.pm/req_cassette/ReqCassette.html)
- **Why replay is needed even at temperature 0**: shared-server batching makes outputs vary (80/1000 unique completions) unless inference is batch-invariant — [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

### Inferences
- Layered cache design for a deterministic LLM compiler:
  - L0: exact-match on (normalized prompt string, parser/model/engine version) -> IR.
  - L1: an optional vCache-style semantic lookup NL -> IR, with an explicit error bound, used only to propose a candidate IR that must still pass round-trip and validation.
  - L2: exact-match on canonical-IR hash -> execution result, which is deterministic by construction.
  - Embedding similarity should never key final outputs directly ("Similarity Is Not Validity"; operational-gap paper).
- APC's keyword-keyed plan templates are effectively a hand-rolled canonical IR. A typed IR with a hash makes this exact and auditable instead of fuzzy.
- Record/replay at the function boundary (cassette-fn, cendor-cassette style) is the right granularity for a compiler: record (IR-hash, model version) -> output. Replay then makes builds reproducible even when the provider's serving stack is not batch-invariant.

### Gaps
- GPTCache: could not retrieve its current status, stars or correctness evaluations (search budget exhausted). vCache's paper positions static-threshold systems like it as lacking guarantees, but that comparison was not read in full.
- No primary source found on Redis/LangChain semantic-cache false-positive rates in production.
- No published study found on caching keyed specifically by a canonical IR or DSL hash. The recommendation above is an inference.
- X/Twitter and Reddit r/LocalLLaMA practitioner discussions could not be searched in this session.
