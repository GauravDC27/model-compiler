"""GCAM v1: the first working version of the Goal-Compiled Agentic MoE (docs/PLAN.md, Part I).

A natural-language goal is parsed into a Goal IR (``parser``), planned into a task graph with
guards and pre-compiled variants (``planner``), scheduled onto calibrated executor ladders by
``dmc.scheduler``, and executed by a recorded runtime (``runtime``) that writes a receipt for
every node. Executors span the plan's assembly: compiled code, classic ML, System-1 encoder and
hub-state heads, the MoE hub with goal-family capacity (goal-gated experts and hot-expert
LoRA), and a specialist spoke. ``evaluate`` measures accuracy and cost against single-model
baselines. On CPU the hub is granite-3.1-1b-a400m and the spoke Qwen2.5-1.5B-Instruct, standing
in for Qwen3.6-35B-A3B and Qwen3.8-27B.
"""
