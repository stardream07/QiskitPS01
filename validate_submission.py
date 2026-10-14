"""Audit all local master-guide requirements and saved scientific evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from qiskit import qpy
from src.problem import build_weighted_graph, cost_diagonal, build_cost_hamiltonian, enumerate_classical
from src.processors import load_processor, BASIS_GATES, NOISE_RATES
from src.qaoa import qaoa_probabilities, build_qaoa_circuit
from src.routing import compile_for_processor, compiled_probabilities, validate_compiled_circuit
from src.metrics import summarize_counts, circuit_metrics
from src.evaluation import exact_noisy_probabilities

ROOT = Path(__file__).resolve().parent


def audit(root=ROOT, require_executed_notebook=True):
    checks = []

    def check(condition, name):
        checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    required = ["README.md", "main.ipynb", "requirements.txt", "report.md", "src/__init__.py", "src/problem.py",
                "src/processors.py", "src/routing.py", "src/metrics.py", "src/visualization.py",
                "processors/processor_A.json", "processors/processor_B.json", "data/raw", "data/processed", "results/tables", "results/figures"]
    for relative in required:
        check((root / relative).exists(), f"Required structure: {relative}")
    submission = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    check(submission["problem_statement"] == "PS-O1", "Exactly one selected problem statement")
    check(2 <= len(submission["team_members"]) <= 6, "Team size is between two and six")
    pins = [line.strip() for line in (root / "requirements.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    check(all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.+-]*", line) for line in pins), "Every direct dependency is pinned")
    check((root / "requirements.lock.txt").is_file(), "Full tested environment lock is included")
    for name in ("README.md", "report.md"):
        text = (root / name).read_text(encoding="utf-8")
        check("[fill" not in text and "TODO" not in text and "PLACEHOLDER" not in text, f"Completed documentation: {name}")
    notebook = nbformat.read(root / "main.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    check(not any(o.output_type == "error" for c in code for o in c.outputs), "Notebook has no saved errors")
    if require_executed_notebook:
        check(all(c.execution_count is not None for c in code), "Every notebook code cell was executed")
    graph = build_weighted_graph()
    _, optimum, _ = enumerate_classical(graph)
    check(optimum == 22, "Exact classical optimum is 22")
    np.testing.assert_allclose(build_cost_hamiltonian(graph).to_matrix(), np.diag(cost_diagonal(graph)), atol=1e-12)
    check(True, "Cost Hamiltonian matches all 32 classical representatives")
    saved_edges = pd.read_csv(root / "data/raw/maxcut_edges.csv")
    check({(int(r.u), int(r.v)): int(r.weight) for r in saved_edges.itertuples()} == {(u, v): d["weight"] for u, v, d in graph.edges(data=True)}, "Saved graph is the reproducible official instance")
    previous_expectation = None
    for depth in (1, 2):
        manifest = json.loads((root / f"results/tables/run_manifest_p{depth}.json").read_text())
        check(manifest["shots_per_evaluation"] == 4096 and manifest["transpiler_seed"] == 42 and manifest["optimization_level"] == 1, f"Official baseline p={depth}")
        check(manifest["basis_gates"] == BASIS_GATES and manifest["noise_rates"] == NOISE_RATES, f"Official common native gates and noise p={depth}")
        for category in ("source_sha256", "artifact_sha256"):
            for relative, digest in manifest[category].items():
                check(hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, f"SHA256 p={depth}: {relative}")
        params = json.loads((root / f"data/processed/qaoa_parameters_p{depth}.json").read_text())
        ref = qaoa_probabilities(cost_diagonal(graph), params["gammas"], params["betas"])
        expectation = float(ref @ cost_diagonal(graph))
        check(np.isclose(expectation, params["best_ideal_expected_cut"], atol=1e-10), f"Recomputed optimized expectation p={depth}")
        check(expectation > cost_diagonal(graph).mean(), f"QAOA beats uniform random mean p={depth}")
        if previous_expectation is not None:
            check(expectation >= previous_expectation, "Warm-started p=2 matches or improves p=1")
        previous_expectation = expectation
        table = pd.read_csv(root / f"results/tables/processor_comparison_p{depth}.csv")
        repetitions = pd.read_csv(root / f"results/tables/sampling_repetitions_p{depth}.csv", dtype={"best_observed_assignment": str})
        check(set(table.processor) == {"Processor A", "Processor B"}, f"Exactly A and B benchmarked p={depth}")
        for name, budget in (("A", 5), ("B", 9)):
            processor = load_processor(root / f"processors/processor_{name}.json")
            check(processor["num_qubits"] == budget, f"Physical budget {name}")
            slug = f"processor_{name.lower()}"
            with (root / f"assets/circuits/{slug}_p{depth}.qpy").open("rb") as handle:
                compiled = qpy.load(handle)[0]
            validate_compiled_circuit(compiled, processor)
            np.testing.assert_allclose(compiled_probabilities(compiled), ref, atol=1e-10)
            check(True, f"Saved native circuit {name} p={depth}: legal coupling and logical equivalence")
            actual, route = compile_for_processor(build_qaoa_circuit(graph, params["gammas"], params["betas"], True), processor)
            check(actual == compiled, f"Circuit recompilation reproducible {name} p={depth}")
            row = table[table.processor == processor["name"]].iloc[0]
            noisy_exact = exact_noisy_probabilities(compiled)
            exact_table = pd.read_csv(root / f"results/tables/{slug}_exact_probabilities_p{depth}.csv")
            np.testing.assert_allclose(noisy_exact, exact_table.noisy_probability_including_readout, atol=1e-10)
            check(np.isclose(noisy_exact @ cost_diagonal(graph), row.exact_noisy_expected_cut, atol=1e-10), f"Exact noisy density-matrix result {name} p={depth}")
            for key in ("circuit_depth", "total_gate_count", "cx_count", "measurement_count"):
                check(row[key] == circuit_metrics(compiled)[key], f"Compiled {key} {name} p={depth}")
            check(row.routing_swap_count == route["routing_swap_count"], f"Measured SWAP routing {name} p={depth}")
            for mode in ("ideal", "noisy"):
                for index, seed in enumerate(manifest["sampling_seeds"]):
                    suffix = "" if index == 0 else f"_seed{seed}"
                    path = root / f"results/tables/{slug}_{mode}_counts_p{depth}{suffix}.csv"
                    counts_table = pd.read_csv(path, dtype={"measurement_key_q4_to_q0": str, "node_bits_0_to_5": str})
                    counts = dict(zip(counts_table.measurement_key_q4_to_q0, map(int, counts_table.shots)))
                    stats, recomputed = summarize_counts(graph, counts, optimum, ref)
                    check(stats["shots"] == 4096, f"4096 shots {name} p={depth} {mode} seed={seed}")
                    recorded = repetitions[(repetitions.processor == processor["name"]) & (repetitions['mode'] == mode) & (repetitions.seed == seed)].iloc[0]
                    check(all(np.isclose(stats[key], recorded[key], atol=1e-10) for key in ("expected_cut_from_samples", "optimal_cut_shot_rate", "approximation_ratio_of_sample_mean", "classical_distribution_fidelity")), f"Recomputed sample metrics {name} p={depth} {mode} seed={seed}")
                    expected_rows = recomputed.sort_values("measurement_key_q4_to_q0").reset_index(drop=True)
                    observed_rows = counts_table.sort_values("measurement_key_q4_to_q0").reset_index(drop=True)
                    check(expected_rows.node_bits_0_to_5.equals(observed_rows.node_bits_0_to_5) and expected_rows.cut_value.equals(observed_rows.cut_value), f"Bit order and cut values {name} p={depth} {mode} seed={seed}")
            check(row.noisy_best_observed_cut == optimum, f"Noisy run sampled an exact optimum {name} p={depth}")
        for figure in ("qaoa_convergence", "processor_comparison", "noisy_cut_distribution", "classical_vs_qaoa_cut"):
            check((root / f"results/figures/{figure}_p{depth}.png").stat().st_size > 1000, f"Exported figure {figure} p={depth}")
    return {"local_requirements_passed": True, "checks_passed": len(checks), "checks": checks,
            "external_requirements": {"repository_url": submission["repository_url"],
                                      "public_repository_final_files_require_upload": True,
                                      "portal_submission_confirmed": submission["portal_submission_confirmed"]},
            "note": "Local validation does not confirm upload to GitHub or submission to the event portal."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unexecuted-notebook", action="store_true")
    args = parser.parse_args()
    result = audit(require_executed_notebook=not args.allow_unexecuted_notebook)
    (ROOT / "results/tables/compliance_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {result['checks_passed']} local checks. GitHub upload and portal submission are separate external steps.")


if __name__ == "__main__":
    main()
