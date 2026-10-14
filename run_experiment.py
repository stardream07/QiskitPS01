"""End-to-end experiment runner for QFF 2026 Phase 1 — PS-O1 Max-Cut.

Run from this project directory:
    python run_experiment.py --classical-only
    python run_experiment.py
    python run_experiment.py --depth 2
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src.problem import build_weighted_graph, save_graph_data, enumerate_classical, enumerate_full_classical
from src.visualization import (
    plot_weighted_graph, plot_processor_graph, plot_cut_comparison,
    plot_convergence, plot_benchmark_bars, plot_cut_distribution,
)
from src.processors import load_processor

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
PROCESSORS = ROOT / "processors"


def ensure_dirs():
    for path in (TABLES, FIGURES, ROOT / "data" / "raw", ROOT / "data" / "processed"):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o)), encoding="utf-8")


def initial_classical_stage(graph):
    save_graph_data(graph, ROOT)
    exact_table, optimum, optimum_assignments = enumerate_classical(graph)
    full_optimum, _full_optimal_assignments = enumerate_full_classical(graph)
    if full_optimum != optimum:
        raise AssertionError(f"Symmetry-reduced optimum {optimum} does not match full optimum {full_optimum}")
    exact_table.to_csv(TABLES / "classical_all_cuts.csv", index=False)
    pd.DataFrame([{
        "classical_optimum_cut_value": optimum,
        "full_64_assignment_optimum_check": full_optimum,
        "symmetry_reduction_verified": bool(full_optimum == optimum),
        "number_of_optimum_representatives_with_node0_fixed": len(optimum_assignments),
        "one_optimal_assignment_node0_to_node5": "".join(map(str, optimum_assignments[0])),
        "graph_seed": 42,
        "weight_seed": 42,
        "symmetry_reduction": "fix node 0 to group 0; cut value is unchanged under complementing all bits",
    }]).to_csv(TABLES / "classical_optimum.csv", index=False)
    plot_weighted_graph(graph, FIGURES / "01_original_weighted_graph.png", assignment=None)
    plot_weighted_graph(graph, FIGURES / "02_classical_optimum_cut.png", assignment=optimum_assignments[0], title=f"Exact classical optimum — cut value {optimum}")
    for filename in ("processor_A.json", "processor_B.json"):
        processor = load_processor(PROCESSORS / filename)
        plot_processor_graph(processor, FIGURES / f"architecture_{filename.replace('.json', '')}.png")
    print("\n[Classical reference]")
    print(f"  Graph edges: {graph.number_of_edges()}")
    print(f"  Exact maximum cut value (32 representatives): {optimum}")
    print(f"  Full 64-assignment verification: {full_optimum} (match: {full_optimum == optimum})")
    print(f"  One optimal assignment (node 0 to node 5): {''.join(map(str, optimum_assignments[0]))}")
    print(f"  Saved graph/processor plots and classical tables under {ROOT}")
    return optimum, optimum_assignments[0]


def run_qaoa_stage(graph, optimum: int, classical_assignment, depth: int, shots: int, seed: int):
    # Lazy import gives users a useful classical-only path before Qiskit is installed.
    from src.qaoa import optimize_qaoa, build_qaoa_circuit
    from src.processors import (
        make_noise_model, transpile_for_processor, get_routing_stage_swap_count,
        TRANSPILE_SEED, NOISE_RATES, BASIS_GATES,
    )
    from src.metrics import circuit_metrics, summarize_counts
    from qiskit_aer import AerSimulator

    print(f"\n[QAOA optimization] depth p={depth}; optimizing exact statevector expectation")
    gammas, betas, best_expected, history, optimizer_info = optimize_qaoa(
        graph, p=depth, seed=seed, restarts=3, maxiter=(100 if depth == 1 else 160)
    )
    print(f"  Best ideal expected cut from statevector: {best_expected:.4f}")
    print(f"  gammas = {np.round(gammas, 5).tolist()}")
    print(f"  betas  = {np.round(betas, 5).tolist()}")
    write_json(ROOT / "data" / "processed" / f"qaoa_parameters_p{depth}.json", {
        "depth_p": depth,
        "gammas": gammas.tolist(),
        "betas": betas.tolist(),
        "best_ideal_expected_cut": best_expected,
        "optimizer": optimizer_info,
        "seed": seed,
    })
    pd.DataFrame({"evaluation": np.arange(1, len(history) + 1), "expected_cut": history}).to_csv(TABLES / f"qaoa_convergence_p{depth}.csv", index=False)
    plot_convergence(history, optimum, FIGURES / f"qaoa_convergence_p{depth}.png", depth)

    logical_measured = build_qaoa_circuit(graph, gammas, betas, measure=True)
    ideal_baseline = __import__("qiskit").transpile(
        logical_measured, basis_gates=BASIS_GATES, optimization_level=1, seed_transpiler=TRANSPILE_SEED
    )
    baseline_cx = int(ideal_baseline.count_ops().get("cx", 0))
    noise_model = make_noise_model()
    result_rows = []
    count_tables = {}
    best_qaoa_assignment = None
    best_qaoa_value = -1

    for filename in ("processor_A.json", "processor_B.json"):
        processor = load_processor(PROCESSORS / filename)
        print(f"  Compiling and simulating {processor['name']}...")
        compiled = transpile_for_processor(logical_measured, processor)
        route_swap_count = get_routing_stage_swap_count(logical_measured, processor)
        metrics = circuit_metrics(compiled)
        routing_cx_delta = int(metrics["cx_count"] - baseline_cx)

        ideal_backend = AerSimulator()
        ideal_counts = ideal_backend.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
        ideal_summary, ideal_count_table = summarize_counts(graph, ideal_counts, optimum)
        ideal_count_table.to_csv(TABLES / f"{processor['name'].lower().replace(' ', '_')}_ideal_counts_p{depth}.csv", index=False)
        count_tables[f"{processor['name']} — ideal"] = ideal_count_table

        noisy_backend = AerSimulator(noise_model=noise_model)
        noisy_counts = noisy_backend.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
        noisy_summary, noisy_count_table = summarize_counts(graph, noisy_counts, optimum)
        noisy_count_table.to_csv(TABLES / f"{processor['name'].lower().replace(' ', '_')}_noisy_counts_p{depth}.csv", index=False)
        count_tables[f"{processor['name']} — noisy"] = noisy_count_table

        candidate_assignment = tuple(int(ch) for ch in noisy_summary["best_observed_assignment"])
        if noisy_summary["best_observed_cut"] > best_qaoa_value:
            best_qaoa_value = noisy_summary["best_observed_cut"]
            best_qaoa_assignment = candidate_assignment

        row = {
            "processor": processor["name"],
            "depth_p": depth,
            "physical_qubits": processor["num_qubits"],
            **metrics,
            "routing_swap_count": int(route_swap_count),
            "unconstrained_baseline_cx_count": baseline_cx,
            "cx_delta_vs_unconstrained_baseline": routing_cx_delta,
            "ideal_expected_cut_from_shots": ideal_summary["expected_cut_from_samples"],
            "ideal_optimal_cut_shot_rate": ideal_summary["optimal_cut_shot_rate"],
            "noisy_expected_cut_from_shots": noisy_summary["expected_cut_from_samples"],
            "noisy_best_observed_cut": noisy_summary["best_observed_cut"],
            "noisy_optimal_cut_shot_rate": noisy_summary["optimal_cut_shot_rate"],
            "noisy_approximation_ratio_of_sample_mean": noisy_summary["approximation_ratio_of_sample_mean"],
            "shots_per_evaluation": shots,
        }
        result_rows.append(row)
        write_json(TABLES / f"{processor['name'].lower().replace(' ', '_')}_summary_p{depth}.json", {
            "processor": processor,
            "metrics": row,
            "ideal_summary": ideal_summary,
            "noisy_summary": noisy_summary,
            "noise_rates": NOISE_RATES,
            "basis_gates": BASIS_GATES,
            "optimization_level": 1,
            "seed_transpiler": TRANSPILE_SEED,
        })
        # Save a portable text summary of the compiled circuit.
        (ROOT / "results" / "tables" / f"{processor['name'].lower().replace(' ', '_')}_compiled_circuit_p{depth}.txt").write_text(str(compiled), encoding="utf-8")

    comparison = pd.DataFrame(result_rows)
    comparison.to_csv(TABLES / f"processor_comparison_p{depth}.csv", index=False)
    plot_benchmark_bars(comparison, FIGURES / f"processor_comparison_p{depth}.png")
    plot_cut_distribution(count_tables, FIGURES / f"noisy_cut_distribution_p{depth}.png")
    if best_qaoa_assignment is not None:
        plot_cut_comparison(
            graph, classical_assignment, best_qaoa_assignment, optimum, best_qaoa_value,
            FIGURES / f"classical_vs_qaoa_cut_p{depth}.png",
        )

    print("\n[Processor comparison]")
    columns = ["processor", "circuit_depth", "cx_count", "routing_swap_count", "ideal_expected_cut_from_shots", "noisy_expected_cut_from_shots", "noisy_optimal_cut_shot_rate"]
    print(comparison[columns].to_string(index=False))
    print(f"\nDone. Tables: {TABLES}\nFigures: {FIGURES}")


def main():
    parser = argparse.ArgumentParser(description="QFF 2026 PS-O1 Max-Cut QAOA benchmark")
    parser.add_argument("--classical-only", action="store_true", help="Generate the graph, exact classical reference and architecture plots without Qiskit")
    parser.add_argument("--depth", type=int, choices=[1, 2], default=1, help="QAOA circuit depth p (official guide permits p=1 or p=2)")
    parser.add_argument("--shots", type=int, default=4096, help="Measurement shots per ideal/noisy processor simulation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for QAOA initialization and simulator sampling")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")

    ensure_dirs()
    graph = build_weighted_graph()
    optimum, classical_assignment = initial_classical_stage(graph)
    if args.classical_only:
        print("\nClassical-only mode complete. Install requirements.txt and rerun without --classical-only for QAOA.")
        return 0
    try:
        run_qaoa_stage(graph, optimum, classical_assignment, depth=args.depth, shots=args.shots, seed=args.seed)
    except ImportError as exc:
        print("\nQAOA stage could not start because a required package is missing.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("Install dependencies from this project's requirements.txt and try again.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\nExperiment failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
