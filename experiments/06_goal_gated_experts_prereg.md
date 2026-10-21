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
