"""Evaluation and analysis scripts (post-training).

**Final entry points**
    - ``evaluate_loop2.py`` — main FMC test / evaluation loop (preset 0: ``data/Test_dataset/``, ``--checkpoint``, ``--test-dir``).
    - ``evaluate_moe2.py`` — MoE evaluation (successor to ``evaluate_moe.py``).

**Naming**
    Scripts with a trailing ``2`` (e.g. ``evaluate_loop2``, ``evaluate_moe2``) are the **final,
    improved** versions; older siblings are kept for backward compatibility.

Other modules: ``evaluate.py``, ``evaluate32.py``, ``analyze_experts.py``, ``analyze_pb.py``.
"""

__all__ = [
    "evaluate",
    "evaluate32",
    "evaluate_loop",
    "evaluate_loop2",
    "evaluate_moe",
    "evaluate_moe2",
    "analyze_experts",
    "analyze_pb",
]
