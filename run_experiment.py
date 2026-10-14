"""Reproduce the complete Phase 1 PS-O1 benchmark (both depths by default)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from src.problem import (build_weighted_graph, save_graph_data, enumerate_classical,
                         enumerate_full_classical, cost_diagonal, build_cost_hamiltonian)
from src.processors import load_processor, BASIS_GATES, NOISE_RATES, TRANSPILE_SEED
from src.visualization import (plot_weighted_graph, plot_processor_graph, plot_cut_comparison,
                               plot_convergence, plot_benchmark_bars, plot_cut_distribution)

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def ensure_dirs(root=ROOT):
    for directory in ("data/raw", "data/processed", "results/tables", "results/figures", "assets/circuits"):
        (root / directory).mkdir(parents=True, exist_ok=True)


def initial_classical_stage(graph, root=ROOT):
    ensure_dirs(root)
    save_graph_data(graph, root)
    table, optimum, assignments = enumerate_classical(graph)
    full_optimum, full_assignments = enumerate_full_classical(graph)
    assert optimum == full_optimum and len(full_assignments) == 2 * len(assignments)
    table.to_csv(root / "results/tables/classical_all_cuts.csv", index=False)
    pd.DataFrame([{"classical_optimum_cut_value": optimum, "full_64_assignment_optimum_check": full_optimum,
                   "symmetry_reduction_verified": True, "number_of_optimum_representatives_with_node0_fixed": len(assignments),
                   "one_optimal_assignment_node0_to_node5": "".join(map(str, assignments[0])),
                   "graph_seed": 42, "weight_seed": 42}]).to_csv(root / "results/tables/classical_optimum.csv", index=False)
    plot_weighted_graph(graph, root / "results/figures/01_original_weighted_graph.png")
    plot_weighted_graph(graph, root / "results/figures/02_classical_optimum_cut.png", assignments[0], f"Exact optimum: cut {optimum}")
    for name in ("A", "B"):
        p = load_processor(ROOT / f"processors/processor_{name}.json")
        plot_processor_graph(p, root / f"results/figures/architecture_processor_{name}.png")
    print(f"Classical reference: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, optimum {optimum}; full enumeration verified.", flush=True)
    return optimum, assignments[0]


def run_qaoa_stage(graph, optimum, classical_assignment, depth=1, shots=4096, seed=42,
                   repeats=5, restarts=16, maxiter=800, warm_start=None, root=ROOT):
    from qiskit import transpile, qpy
    from qiskit_aer import AerSimulator
    from src.qaoa import optimize_qaoa, build_qaoa_circuit, qaoa_probabilities
    from src.routing import compile_for_processor, compiled_probabilities
    from src.metrics import circuit_metrics, summarize_counts
    from src.processors import make_noise_model
    from src.evaluation import exact_noisy_probabilities

    ensure_dirs(root)
    tables, figures = root / "results/tables", root / "results/figures"
    print(f"Optimizing QAOA p={depth}: {restarts} starts, up to {maxiter} COBYLA evaluations/start...", flush=True)
    if depth == 2 and warm_start is None:
        g1, b1, *_ = optimize_qaoa(graph, 1, seed, restarts, maxiter)
        warm_start = (g1, b1)
    gammas, betas, expectation, history, optimizer = optimize_qaoa(graph, depth, seed, restarts, maxiter, warm_start)
    record = {"depth_p": depth, "gammas": gammas.tolist(), "betas": betas.tolist(),
              "best_ideal_expected_cut": expectation, "optimizer": optimizer, "seed": seed}
    write_json(root / f"data/processed/qaoa_parameters_p{depth}.json", record)
    write_json(root / "data/processed/cost_hamiltonian.json", {"num_qubits": 5, "pauli_terms": build_cost_hamiltonian(graph).to_list(),
               "diagonal": cost_diagonal(graph).tolist(), "cost_unitary": "exp(-i*gamma*C), global identity phase omitted"})
    pd.DataFrame({"evaluation": np.arange(1, len(history) + 1), "expected_cut": history,
                  "best_so_far": np.maximum.accumulate(history)}).to_csv(tables / f"qaoa_convergence_p{depth}.csv", index=False)
    plot_convergence(history, optimum, figures / f"qaoa_convergence_p{depth}.png", depth)
    reference = qaoa_probabilities(cost_diagonal(graph), gammas, betas)
    pd.DataFrame({"measurement_key_q4_to_q0": [format(i, "05b") for i in range(32)],
                  "probability": reference, "cut_value": cost_diagonal(graph)}).to_csv(tables / f"logical_probabilities_p{depth}.csv", index=False)
    logical = build_qaoa_circuit(graph, gammas, betas, measure=True)
    with (root / f"assets/circuits/logical_p{depth}.qpy").open("wb") as handle:
        qpy.dump(logical, handle)
    baseline = transpile(logical, basis_gates=BASIS_GATES, optimization_level=1, seed_transpiler=TRANSPILE_SEED)
    baseline_cx = int(baseline.count_ops().get("cx", 0))
    ideal_backend = AerSimulator(method="statevector", max_parallel_threads=1)
    noisy_backend = AerSimulator(method="density_matrix", noise_model=make_noise_model(), max_parallel_threads=1)
    rows, repetition_rows, count_tables = [], [], {}
    best_assignment, best_cut = None, -1
    for name in ("A", "B"):
        processor = load_processor(ROOT / f"processors/processor_{name}.json")
        slug = f"processor_{name.lower()}"
        print(f"  Compiling {processor['name']} and checking logical equivalence...", flush=True)
        compiled, routing = compile_for_processor(logical, processor)
        exact = compiled_probabilities(compiled)
        error = float(np.max(np.abs(exact - reference)))
        if error > 1e-10:
            raise AssertionError(f"{processor['name']} changed the ideal logical distribution: {error}")
        metrics = circuit_metrics(compiled)
        exact_noisy = exact_noisy_probabilities(compiled)
        exact_noisy_mean = float(exact_noisy @ cost_diagonal(graph))
        pd.DataFrame({"measurement_key_q4_to_q0": [format(i, "05b") for i in range(32)],
                      "ideal_probability": exact, "noisy_probability_including_readout": exact_noisy,
                      "cut_value": cost_diagonal(graph)}).to_csv(tables / f"{slug}_exact_probabilities_p{depth}.csv", index=False)
        with (root / f"assets/circuits/{slug}_p{depth}.qpy").open("wb") as handle:
            qpy.dump(compiled, handle)
        (tables / f"{slug}_compiled_circuit_p{depth}.txt").write_text(str(compiled.draw(output="text", fold=120)), encoding="utf-8")
        summaries = {}
        # Each repetition still uses the guide's 4096-shot evaluation baseline.
        for mode, backend in (("ideal", ideal_backend), ("noisy", noisy_backend)):
            for repetition in range(repeats):
                sample_seed = seed + repetition
                counts = backend.run(compiled, shots=shots, seed_simulator=sample_seed).result().get_counts()
                summary, table = summarize_counts(graph, counts, optimum, reference)
                assert summary["shots"] == shots
                suffix = "" if repetition == 0 else f"_seed{sample_seed}"
                table.to_csv(tables / f"{slug}_{mode}_counts_p{depth}{suffix}.csv", index=False)
                repetition_rows.append({"processor": processor["name"], "depth_p": depth, "mode": mode,
                                        "seed": sample_seed, **summary})
                if repetition == 0:
                    summaries[mode] = summary
                    count_tables[f"{processor['name']} - {mode}"] = table
                    if mode == "noisy" and summary["best_observed_cut"] > best_cut:
                        best_cut = summary["best_observed_cut"]
                        best_assignment = tuple(map(int, summary["best_observed_assignment"]))
        ideal, noisy = summaries["ideal"], summaries["noisy"]
        row = {"processor": processor["name"], "depth_p": depth, "physical_qubits": processor["num_qubits"],
               **metrics, "routing_swap_count": routing["routing_swap_count"],
               "unconstrained_baseline_cx_count": baseline_cx,
               "cx_delta_vs_unconstrained_baseline": metrics["cx_count"] - baseline_cx,
               "compiled_probability_max_abs_error": error, "exact_logical_expected_cut": expectation,
               "exact_logical_approximation_ratio": expectation / optimum,
               "ideal_expected_cut_from_shots": ideal["expected_cut_from_samples"],
               "ideal_optimal_cut_shot_rate": ideal["optimal_cut_shot_rate"],
               "noisy_expected_cut_from_shots": noisy["expected_cut_from_samples"],
               "noisy_cut_mean_standard_error": noisy["cut_mean_standard_error"],
               "noisy_cut_mean_ci95_low": noisy["cut_mean_ci95_low"], "noisy_cut_mean_ci95_high": noisy["cut_mean_ci95_high"],
               "noisy_best_observed_cut": noisy["best_observed_cut"],
               "exact_noisy_expected_cut": exact_noisy_mean,
               "exact_noisy_approximation_ratio": exact_noisy_mean / optimum,
               "exact_noisy_optimal_cut_probability": float(exact_noisy[cost_diagonal(graph) == optimum].sum()),
               "exact_noisy_classical_distribution_fidelity": float(np.sqrt(exact_noisy * reference).sum() ** 2),
               "noisy_optimal_cut_shot_rate": noisy["optimal_cut_shot_rate"],
               "noisy_approximation_ratio_of_sample_mean": noisy["approximation_ratio_of_sample_mean"],
               "noisy_classical_distribution_fidelity": noisy["classical_distribution_fidelity"],
               "noisy_total_variation_distance": noisy["total_variation_distance"],
               "shots_per_evaluation": shots, "primary_sample_seed": seed, "repetitions": repeats}
        rows.append(row)
        write_json(tables / f"{slug}_summary_p{depth}.json", {"processor": processor, "metrics": row, "routing": routing,
                   "ideal_summary": ideal, "noisy_summary": noisy, "noise_rates": NOISE_RATES, "basis_gates": BASIS_GATES,
                   "exact_noisy_distribution_method": "Aer density matrix plus independent 1% readout confusion on each measured bit",
                   "optimization_level": 1, "seed_transpiler": TRANSPILE_SEED,
                   "distribution_fidelity_definition": "(sum sqrt(p_exact * p_observed))**2; classical, not quantum state fidelity"})
        print(f"  {processor['name']}: depth {metrics['circuit_depth']}, CX {metrics['cx_count']}, SWAPs {routing['routing_swap_count']}, noisy mean {noisy['expected_cut_from_samples']:.4f}", flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables / f"processor_comparison_p{depth}.csv", index=False)
    pd.DataFrame(repetition_rows).to_csv(tables / f"sampling_repetitions_p{depth}.csv", index=False)
    plot_benchmark_bars(comparison, figures / f"processor_comparison_p{depth}.png")
    plot_cut_distribution(count_tables, figures / f"noisy_cut_distribution_p{depth}.png")
    plot_cut_comparison(graph, classical_assignment, best_assignment, optimum, best_cut, figures / f"classical_vs_qaoa_cut_p{depth}.png")
    packages = ["numpy", "pandas", "matplotlib", "networkx", "scipy", "qiskit", "qiskit-aer", "nbformat", "nbclient", "ipykernel", "pytest"]
    sources = list((ROOT / "src").glob("*.py")) + [ROOT / "run_experiment.py", ROOT / "requirements.txt"] + list((ROOT / "processors").glob("*.json"))
    output_files = list(tables.glob(f"*p{depth}*")) + list(figures.glob(f"*p{depth}*")) + list((root / "assets/circuits").glob(f"*p{depth}*"))
    output_files += [root / f"data/processed/qaoa_parameters_p{depth}.json", root / "data/processed/maxcut_graph.json", root / "data/raw/maxcut_edges.csv", root / "data/processed/cost_hamiltonian.json"]
    manifest = {"problem_statement": "PS-O1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "package_versions": {p: metadata.version(p) for p in packages}, "depth_p": depth,
                "shots_per_evaluation": shots, "sampling_seeds": list(range(seed, seed + repeats)),
                "optimizer_seed": seed, "graph_seed": 42, "weight_seed": 42, "transpiler_seed": 42,
                "optimization_level": 1, "basis_gates": BASIS_GATES, "noise_rates": NOISE_RATES,
                "command": f"python run_experiment.py --depth {depth} --shots {shots} --seed {seed} --repeats {repeats} --restarts {restarts} --maxiter {maxiter}",
                "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "artifact_sha256": {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in output_files if not p.name.startswith('run_manifest')}}
    write_json(tables / f"run_manifest_p{depth}.json", manifest)
    print(f"Completed p={depth}; exact optimized mean {expectation:.6f}, ratio {expectation / optimum:.6f}.", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--depth", choices=["1", "2", "both"], default="both")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if min(args.shots, args.repeats, args.restarts, args.maxiter) < 1 or args.seed < 0:
        parser.error("Shot/repetition/optimization budgets must be positive and seed nonnegative")
    root = args.output_dir.resolve()
    graph = build_weighted_graph()
    optimum, assignment = initial_classical_stage(graph, root)
    if args.classical_only:
        return 0
    previous = None
    for depth in ([1, 2] if args.depth == "both" else [int(args.depth)]):
        warm = (previous["gammas"], previous["betas"]) if previous is not None and depth == 2 else None
        previous = run_qaoa_stage(graph, optimum, assignment, depth, args.shots, args.seed, args.repeats,
                                  args.restarts, args.maxiter, warm, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
