"""Circuit and sample metrics for Max-Cut benchmarking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .problem import cut_value, full_assignment_from_reduced


def circuit_metrics(circuit) -> dict:
    ops = circuit.count_ops()
    quantum_ops = {name: count for name, count in ops.items() if name not in {"measure", "barrier"}}
    return {
        "circuit_depth": int(circuit.depth() or 0),
        "total_gate_count": int(sum(quantum_ops.values())),
        "measurement_count": int(ops.get("measure", 0)),
        "unitary_depth": int(circuit.remove_final_measurements(inplace=False).depth() or 0),
        "cx_count": int(ops.get("cx", 0)),
        "swap_count_after_native_translation": int(ops.get("swap", 0)),
        "single_qubit_rz_count": int(ops.get("rz", 0)),
        "single_qubit_sx_count": int(ops.get("sx", 0)),
        "single_qubit_x_count": int(ops.get("x", 0)),
    }


def measurement_key_to_assignment(key: str) -> tuple[int, ...]:
    """Convert Qiskit count key (c4...c0) into graph-node bits [node0..node5]."""
    clean = key.replace(" ", "")
    if len(clean) != 5 or any(bit not in "01" for bit in clean):
        raise ValueError("Expected a five-bit binary Qiskit measurement key")
    qbits = tuple(int(character) for character in clean[::-1])
    return full_assignment_from_reduced(qbits)


def summarize_counts(graph, counts: dict[str, int], classical_optimum: int, reference_probabilities=None) -> tuple[dict, pd.DataFrame]:
    if not counts or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0 for v in counts.values()):
        raise ValueError("Counts must be nonnegative integers in a nonempty histogram")
    total_shots = int(sum(counts.values()))
    if total_shots < 1 or classical_optimum <= 0:
        raise ValueError("Shot total and classical optimum must be positive")
    rows = []
    best_observed = -1
    best_observed_bits = None
    expected_cut = 0.0
    feasible_count = 0
    optimum_count = 0
    for key, count in sorted(counts.items()):
        if count == 0:
            continue
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
    mean = summary["expected_cut_from_samples"]
    variance = sum(row["shots"] * (row["cut_value"] - mean) ** 2 for row in rows) / max(1, total_shots - 1)
    se = float(np.sqrt(variance / total_shots))
    summary.update({"cut_mean_standard_error": se,
                    "cut_mean_ci95_low": max(0.0, mean - 1.96 * se),
                    "cut_mean_ci95_high": min(float(classical_optimum), mean + 1.96 * se),
                    "approximation_ratio_of_best_observed": best_observed / classical_optimum})
    # Wilson score interval for the optimal-cut probability.
    rate, z, n = optimum_count / total_shots, 1.96, total_shots
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    summary.update({"optimal_cut_rate_ci95_low": max(0.0, float(center - half)),
                    "optimal_cut_rate_ci95_high": min(1.0, float(center + half))})
    if reference_probabilities is not None:
        ref = np.asarray(reference_probabilities, dtype=float)
        if ref.shape != (32,) or np.any(ref < 0) or not np.isclose(ref.sum(), 1):
            raise ValueError("Reference distribution must be 32 normalized probabilities")
        observed = np.zeros(32)
        for key, count in counts.items():
            observed[int(key.replace(" ", ""), 2)] = count / total_shots
        summary.update({"classical_distribution_fidelity": float(np.square(np.sqrt(observed * ref).sum())),
                        "total_variation_distance": float(np.abs(observed - ref).sum() / 2)})
    table = pd.DataFrame(rows).sort_values(["cut_value", "shots"], ascending=[False, False]).reset_index(drop=True)
    return summary, table
