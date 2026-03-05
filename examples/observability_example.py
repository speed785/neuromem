import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from neuromem import ContextManager
from neuromem.observability import export_prometheus, get_metrics, reset_metrics


def run() -> None:
    reset_metrics()
    cm = ContextManager(token_budget=120, always_keep_last_n=1, debug=True)
    cm.add_system("You are a strict assistant. Keep critical requirements.")

    for i in range(10):
        cm.add_user(f"User turn {i}: this is a critical requirement that must not be lost.")
        cm.add_assistant("Acknowledged. I will preserve this requirement across pruning.")

    _ = cm.get_messages(force_prune=True)
    metrics = get_metrics()

    print("Metrics snapshot:")
    print(metrics)
    print()
    print("Prometheus export:")
    print(export_prometheus())


if __name__ == "__main__":
    run()
