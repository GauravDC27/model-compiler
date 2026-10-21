# Pre-registration — goal-gated expert expansion on granite-3.1-1b-a400m (plan v5 item ii)

Registered 2026-10-21, before any run.

**Question.** Can new experts added to a pretrained MoE, with router rows trained to send a
goal's traffic to them, raise accuracy on that goal while leaving every other goal untouched?

**Model.** granite-3.1-1b-a400m-instruct (24 MoE layers, 32 experts, top-8), FP32, CPU, base frozen.

**Goal and data.** Goal `decide.oracle`: yes/no decisions from `dmc.oracle` (6 families:
compare, even, refund, letter, sum, expense; 3 paraphrase templates each). Train:
`generate(150, seed=100)` (2,700 items). Test: `generate(20, seed=0)` (360 items; the 03 set).
Training target: the gold label as the assistant's first token after the Decide-profile prompt
(canonical label order); loss on that token only.

**Arms** (one new expert per layer in all 24 layers; trainable = new experts + router rows +
goal bias; AdamW lr 1e-3, 200 steps, batch 8, seed 0):
- `routed` — new expert competes in top-8 via `w_new·h + b_goal` (b_goal init 0); warm-started
  as a copy of the layer's most-selected base expert on the training prompts ("discovered" expert).
- `additive` — base top-8 untouched; new expert added with gate `sigmoid(w_new·h + b_goal)`
  (b_goal init 0), down-projection zero-initialized (exact no-op at step 0).

**Measures** (Decide profile, R17 scorer, abstain margin 0):
1. Oracle accuracy on the 360 test decisions with the goal on: base vs each arm; paired
   difference with a group-level bootstrap 95% CI (resampling paraphrase groups).
2. Goal-off identity: logits on 20 probe prompts with the goal off are bit-identical to the
   base (`torch.equal`).
3. Collateral drift with the goal on: wikitext-2 validation CE (16 chunks) vs base.
4. New-expert traffic share with the goal on, averaged over all expanded layers using each
   layer's actual input (routed: share of tokens with a new expert in top-8; additive: mean
   gate), on 30 test prompts and on wikitext.
5. Cost: trainable parameter count; CPU forward time goal-on vs goal-off.

**Success (per arm):** accuracy gain ≥ +10 points with CI lower bound > 0; goal-off identity
holds; goal-on wikitext CE rise ≤ 5%. Generalization note: test items come from a different
seed (new numbers/words), same templates; a template-held-out split is out of scope here.

---

**Amendment 1, registered 2026-10-21, before any run** (round-4 review M8).

- *Goal split and wrong-goal test.* The goal `numeric` is trained on the families {compare, even,
  sum, expense} only (training items from `generate(150, seed=100)` restricted to them). Test:
  those families' items in `generate(20, seed=0)` (240 decisions). **Wrong-goal test:** the
  held-out families {refund, letter} (120 decisions) evaluated *with the goal on*, against the base.
- *Arms* (all 150 steps, batch 8, AdamW lr 1e-3, seed 0; one new expert per layer in all 24
  layers where applicable):
  1. `routed` — parent = the layer's most-selected base expert on the goal's training prompts;
     the copy and its router row (initialized from the parent's row) both get Gaussian noise of
     0.05 × their std; initial goal bias +0.1 (so the copy is preferred over its identical
     parent at the start instead of losing the lowest-index tie-break). The **parent itself is
     not perturbed**, unlike UMoE, because the base must stay bit-identical with the goal off;
     this deviation from the plan's "noise on both" is deliberate and will be stated in the plan.
  2. `additive` — as registered originally (zero down-projection, zero router row, bias 0).
  3. `router_only` — as `routed` but with an exact (noise-free) parent copy, and only the router
     row and goal bias are trained (experts frozen).
  4. `random_parent` — as `routed` but the parent is a random base expert (seeded).
  5. `lora` — ordinary fine-tuning baseline (item i): rank-r LoRA on each layer's hot experts
     (80% of selection mass on the goal's training prompts), router untouched, r chosen so the
     trainable-parameter count matches `routed` as closely as an integer rank allows.
- *Primary comparison:* each new-capacity arm (`routed`, `additive`) vs `lora` on goal accuracy,
  paired group-bootstrap 95% CI. **Success for an arm:** CI lower bound > 0 vs `lora`, goal-off
  bit-identity holds, goal-on wikitext CE rise ≤ 5%. Gains vs the base are reported but are not
  the test (the base is near chance on several families, so any supervision clears +10 points).
- *Secondary:* wrong-goal accuracy change vs base on held-out families (with CI); new-expert
  traffic share on goal prompts vs wikitext; `router_only` and `random_parent` vs `routed`
  (does training the expert matter; does the discovered parent matter).
- *Scope:* in-template generalization only (same templates, new numbers/words); this cannot
  speak to production-scale claims about item (ii).

**Amendment 2, registered 2026-10-21, before any run** (round-5 review N14).

- *Multiplicity.* There are two primary comparisons (`routed` vs `lora`, `additive` vs `lora`), so
  an arm's success uses a Bonferroni-adjusted **97.5%** paired group-bootstrap CI (lower bound > 0),
  plus goal-off bit-identity and goal-on wikitext CE rise ≤ 5%. Item (ii) is supported if at least
  one of the two arms succeeds. 95% CIs are still reported.
- *Interpretation rule for the controls* (fixed now). `router_only` and `random_parent` get the
  same comparison against `lora` (reported as `CLEARS_LORA_BAR`). If a successful arm's result is
  matched by `random_parent` (it also clears the bar), the pilot supports "new capacity helps" but
  not "the discovered parent matters". If `router_only` also clears the bar, the pilot supports
  neither claim about trained expert capacity: routing alone explains the gain. Only if `routed`
  clears the bar and both controls do not does the pilot support the full item (ii) mechanism.
