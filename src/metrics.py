"""Circuit and sample metrics for Max-Cut benchmarking."""
from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd

from .problem import cut_value, full_assignment_from_reduced


def circuit_metrics(circuit) -> dict:
    ops = circuit.count_ops()
    return {
        "circuit_depth": int(circuit.depth() or 0),
        "total_gate_count": int(sum(ops.values())),
        "cx_count": int(ops.get("cx", 0)),
        "swap_count_after_native_translation": int(ops.get("swap", 0)),
        "single_qubit_rz_count": int(ops.get("rz", 0)),
        "single_qubit_sx_count": int(ops.get("sx", 0)),
        "single_qubit_x_count": int(ops.get("x", 0)),
    }


def measurement_key_to_assignment(key: str) -> tuple[int, ...]:
    """Convert Qiskit count key (c4...c0) into graph-node bits [node0..node5]."""
    clean = key.replace(" ", "")
    qbits = tuple(int(character) for character in clean[::-1])
    return full_assignment_from_reduced(qbits)


def summarize_counts(graph, counts: dict[str, int], classical_optimum: int) -> tuple[dict, pd.DataFrame]:
    total_shots = int(sum(counts.values()))
    rows = []
    best_observed = -1
    best_observed_bits = None
    expected_cut = 0.0
    feasible_count = 0
    optimum_count = 0
    for key, count in counts.items():
        assignment = measurement_key_to_assignment(key)
        value = cut_value(graph, assignment)
        expected_cut += value * int(count)
        if value > best_observed:
            best_observed = value
            best_observed_bits = "".join(map(str, assignment))
        rows.append({
            "measurement_key_q4_to_q0": key,
            "node_bits_0_to_5": "".join(map(str, assignment)),
            "cut_value": int(value),
            "shots": int(count),
            "frequency": int(count) / total_shots if total_shots else 0.0,
        })
        # For Max-Cut there are no hard feasibility constraints in this reduced
        # representation; every bitstring corresponds to a valid cut.
        feasible_count += int(count)
        if value == classical_optimum:
            optimum_count += int(count)
    summary = {
        "shots": total_shots,
        "expected_cut_from_samples": expected_cut / total_shots if total_shots else float("nan"),
        "best_observed_cut": int(best_observed),
        "best_observed_assignment": best_observed_bits or "",
        "classical_optimum": int(classical_optimum),
        "optimal_cut_shot_rate": optimum_count / total_shots if total_shots else 0.0,
        "feasible_shot_rate": feasible_count / total_shots if total_shots else 0.0,
        "approximation_ratio_of_sample_mean": (expected_cut / total_shots / classical_optimum) if total_shots and classical_optimum else float("nan"),
    }
    table = pd.DataFrame(rows).sort_values(["cut_value", "shots"], ascending=[False, False]).reset_index(drop=True)
    return summary, table
