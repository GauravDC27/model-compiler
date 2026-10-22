"""dmc — Deterministic Model Compiler.

Reference implementation of the compiler-style pipeline described in docs/PLAN.md:
natural-language prompt -> typed IR -> static dispatch -> specialist execution profile
-> verification, with every step content-addressed and replayable.
"""

__version__ = "0.1.0"
