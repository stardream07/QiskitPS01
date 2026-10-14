"""Generate documentation and the runnable notebook from measured outputs.

Run after run_experiment.py. Then execute_notebook.py and validate_submission.py.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from src.processors import load_processor

ROOT = Path(__file__).resolve().parent


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    tables = {p: pd.read_csv(ROOT / f"results/tables/processor_comparison_p{p}.csv") for p in (1, 2)}
    params = {p: json.loads((ROOT / f"data/processed/qaoa_parameters_p{p}.json").read_text()) for p in (1, 2)}
    manifest = json.loads((ROOT / "results/tables/run_manifest_p2.json").read_text())
    settings = {json.loads((ROOT / f"results/tables/run_manifest_p{p}.json").read_text())["shots_per_evaluation"] for p in (1, 2)}
    if settings != {4096}:
        raise ValueError("Build the final guide-compliant documentation from 4096-shot runs")
    key_rows = []
    for p, table in tables.items():
        for row in table.itertuples():
            key_rows.append([p, row.processor, row.circuit_depth, row.cx_count, row.routing_swap_count,
                             f"{row.exact_noisy_expected_cut:.4f}", f"{row.exact_noisy_approximation_ratio:.4f}",
                             f"{100 * row.exact_noisy_optimal_cut_probability:.2f}%"])
    summary_table = md_table(["p", "Processor", "Depth", "CX", "Routing SWAPs", "Exact noisy mean cut", "Mean ratio", "P(optimum)"], key_rows)
    topology_rows = []
    for name in ("A", "B"):
        processor = load_processor(ROOT / f"processors/processor_{name}.json")
        graph = nx.Graph()
        graph.add_nodes_from(range(processor["num_qubits"]))
        graph.add_edges_from(processor["edges"])
        topology_rows.append([processor["name"], len(graph), graph.number_of_edges(), nx.diameter(graph),
                              max(dict(graph.degree()).values()), f"{nx.density(graph):.3f}"])
    topology_table = md_table(["Processor", "Physical qubits", "Undirected edges", "Diameter", "Maximum degree", "Density"], topology_rows)

    readme = f'''# Qiskit Fall Fest 2026 - Phase 1, PS-O1

**Selected problem:** PS-O1, Max-Cut on Weighted Graphs.
**Team members:** Inderpreet Kaur and Harshit Tripathi.
**Repository:** https://github.com/stardream07/QiskitPS01

This project solves the guide's six-node weighted graph with QAOA at both allowed depths, p=1 and p=2, and compares participant-defined Processor A (5 qubits) and Processor B (9 qubits). Exact enumeration gives cut **22**. All four primary noisy runs sampled this optimum; average quality is reported separately from the best sample.

## Measured results

{summary_table}

Depth includes final measurement. SWAPs are captured after routing in the **same** pipeline that produces the simulated circuit. The exact noisy means use density matrices and readout noise; each processor also has five 4096-shot ideal/noisy evaluations (seeds 42-46), saved separately. The ratio is mean cut / 22. p=2's exact ideal mean is **{params[2]['best_ideal_expected_cut']:.6f}**, ratio **{params[2]['best_ideal_expected_cut']/22:.6f}**. A best observed cut of 22 does not imply every shot is optimal.

![Processor comparison](results/figures/processor_comparison_p2.png)

## Setup

The final package was executed and tested with **Python {manifest['python']} on Windows**, Qiskit 2.5.0 and Aer 0.17.2. Use Python 3.14 for the supplied full environment lock.

```powershell
py -3.14 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.lock.txt
.\\.venv\\Scripts\\python.exe -m pip check
.\\.venv\\Scripts\\python.exe run_experiment.py
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe execute_notebook.py
.\\.venv\\Scripts\\python.exe validate_submission.py
```

On Linux/macOS, replace the executable with `.venv/bin/python`, creating the environment with `python3.14 -m venv .venv`. Direct dependencies are pinned in `requirements.txt`; `requirements.lock.txt` also pins their transitive dependencies. Select the local `.venv` kernel when opening `main.ipynb` interactively. The saved notebook contains a complete successful run.

## Methodology and fairness

1. Generate `networkx.erdos_renyi_graph(n=6, p=0.5, seed=42)`. Assign integer weights 1-5 using `numpy.random.default_rng(42)` in sorted edge order. The guide leaves the weight seed unspecified; 42 is our documented choice.
2. Fix node 0 to bit 0 using global complement symmetry. Every full cut has an equal-value complement; this retains all cut values and the optimum while requiring five logical qubits. Verify all 64 full assignments against the 32 representatives. This reduction defines a five-qubit QAOA ansatz; it does not claim to reproduce the finite-depth state of an unreduced six-qubit mixer.
3. Build the cost Hamiltonian `C = sum w*(1-Zu*Zv)/2`, with `Z0=+1`. Use `RZ(-gamma*w)`, `RZZ(-gamma*w)`, and `RX(2*beta)`. Optimize exact expected cut with grid-seeded, 16-start COBYLA for p=1 and 16-start warm/random COBYLA for p=2. Verify the fast NumPy simulator against Qiskit Statevector.
4. Hold angles and identity initial layout fixed across A/B. Compile at optimization level **1**, transpiler seed **42**, basis **{{rz, sx, x, cx}}**. Check every physical CX edge and the routed measurement distribution. Capture SWAPs with an analysis-only post-routing hook.
5. Apply the same synthetic noise on both: one-qubit depolarizing **0.001** on rz/sx/x, CX depolarizing **0.005**, symmetric readout flips **0.01**. RZ is deliberately noisy under this synthetic model. No device calibration or real hardware is claimed.
6. Compare mean and best cut, mean approximation ratio, optimal-cut probability, circuit depth/CX/SWAP counts, and classical distribution fidelity. Report confidence intervals and five sampling repetitions.

## Architecture design

{topology_table}

Processor A uses the guide's linear-chain reference. Processor B is a connected 12-edge graph whose five-qubit core exactly contains the six interactions of this reduced instance; the four remaining physical nodes provide a sparse extension. B is a deliberately problem-specific co-design, and extra qubit count alone is not its explanation. JSON files include the connectivity, baseline configuration, noise rates and rationale. The guide permits participant-designed maps; it does not require a heavy-hex map or exactly 12 edges.

## Files and reproduction

```text
README.md                         setup, method and key measured results
main.ipynb                        executed, end-to-end notebook
requirements.txt                  pinned direct dependencies
requirements.lock.txt             full tested dependency lock
run_experiment.py                  regenerate both depths and all outputs
execute_notebook.py                execute using a fresh local kernel
validate_submission.py             audit guide requirements and evidence
build_deliverables.py              rebuild this documentation and notebook
submission.json                   selected problem, members and portal status
data/raw/                         generated weighted edge list
data/processed/                   graph, Hamiltonian and optimized angles
processors/processor_A.json        5-qubit model and benchmark config
processors/processor_B.json        9-qubit model and benchmark config
src/problem.py                    objective, Hamiltonian, classical references
src/processors.py                 topology validation and noise model
src/routing.py                    transpilation, SWAP capture and layout checks
src/qaoa.py                        circuit and parameter optimization
src/evaluation.py                  exact noisy measurement distributions
src/metrics.py                     resource, solution and fidelity metrics
src/visualization.py               saved scientific plots
results/tables/                    CSV/JSON results, test and audit evidence
results/figures/                   graphs, coupling maps and comparison plots
assets/circuits/                   portable QPY logical and physical circuits
tests/                            independent scientific correctness tests
report.md                         completed architectural analysis
```

`python run_experiment.py --depth 1` and `--depth 2` run individual depths; `--classical-only` regenerates the classical reference. `--shots`, `--seed`, `--repeats`, `--restarts` and `--maxiter` support exploratory runs; keep default settings for the final guide baseline. `--output-dir PATH` writes an experiment into a separate folder. After changing scientific settings, regenerate documentation and execute the notebook before the final audit. GitHub Actions runs tests, the notebook and the audit in a fresh Windows environment.

Each depth has a `run_manifest_p*.json` recording versions, seeds, command, source hashes and artifact hashes. Count keys are Qiskit `c4...c0`; graph assignments in exports are `node0...node5`. Source hashes detect stale results. Cross-machine floating-point or compiler changes can cause slight variation; the lock and manifests record the tested environment.

## Limitations and submission

The graph is small enough to solve exactly classically. This is an architecture experiment, with no quantum speedup claim. Low-depth QAOA and multistart search do not guarantee a global variational optimum. Processor B is tailored to one graph; synthetic independent noise omits relaxation, crosstalk, timing and calibration drift. Distribution fidelity is a **classical measurement-distribution** metric, not quantum state fidelity.

The local master-guide requirements are audited in `results/tables/compliance_check.json`. The final files must be present in the public repository and its URL submitted at https://qffsrmap2026.com/learning/hackathon by **October 14, 2026, 23:59 IST**. Upload and portal acceptance are external steps; this repository does not assert they have happened. Processor C/D are outside Phase 1.

Read [report.md](report.md) for the full quantitative analysis.
'''
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    report = f'''# PS-O1: Weighted Max-Cut QAOA and processor co-design

**Qiskit Fall Fest 2026, Phase 1**
**Authors:** Inderpreet Kaur and Harshit Tripathi.
**Problem:** exactly PS-O1. All results below come from the included executable code.

## 1. Problem and classical reference

The objective is to maximize crossing edge weight on the six-node NetworkX graph generated with `n=6`, `p=0.5`, and topology seed 42. Weights use NumPy `default_rng(42)`, sampled as integers from 1 to 5 in sorted edge order; this seed is an explicit choice because the guide does not specify it. The nine edges have total weight 27, so uniform independent partitions have expected cut 13.5.

The exact optimum is **22**. A representative partition is node bits **001011**, i.e. group 0 = {{0,1,3}} and group 1 = {{2,4,5}}. Its crossing weights are 1+4+3+3+5+4+2=22. Both all-64 full assignments and all-32 reduced assignments are enumerated; exported tables retain every reduced cut.

![Weighted graph](results/figures/01_original_weighted_graph.png)
![Classical solution](results/figures/02_classical_optimum_cut.png)

## 2. Five-qubit formulation and QAOA

Complementing all six bits leaves every crossing edge unchanged. Fixing node 0 to 0 selects exactly one representative of each complementary pair and preserves the full set of attainable cut values. Logical `q0...q4` encode nodes 1...5. This objective-preserving reduction fits the five-qubit budget; its free-variable mixer defines a reduced ansatz, whose finite-p distribution need not equal an unreduced six-qubit QAOA distribution.

The diagonal cost Hamiltonian is

$$C=\\sum_{{(u,v)\\in E}}\\frac{{w_{{uv}}}}{{2}}(I-Z_uZ_v),\\quad Z_0=+1.$$

For an edge touching node 0 the interaction becomes a single Z. The stored `SparsePauliOp` retains the identity offset, so its diagonal matches classical cut values exactly. Starting from the uniform superposition, each QAOA layer uses `exp(-i gamma C)` and `exp(-i beta sum X)`. Omitting the irrelevant identity phase yields `RZ(-gamma*w)` and `RZZ(-gamma*w)`; the mixer is `RX(2*beta)`. A test explicitly checks the sign convention against diagonal phase evolution.

For p=1, a 65-by-41 gamma/beta grid across `[0,2*pi)` and `[0,pi)` supplies separated starting points. COBYLA refines 16 starts with up to 800 evaluations each and tolerance 1e-7. p=2 uses the optimized p=1 state as a feasible starting point, a split-layer start, and seeded random starts. Every objective evaluation is saved and the best evaluated angles are retained, including when a local run stops at its budget. The fast NumPy objective is independently verified against Qiskit Statevector. This search makes no global variational optimality claim.

'''
    optimizer_rows = []
    for p in (1, 2):
        q = params[p]
        optimizer_rows.append([p, ", ".join(f"{a:.7f}" for a in q["gammas"]), ", ".join(f"{a:.7f}" for a in q["betas"]),
                               f"{q['best_ideal_expected_cut']:.6f}", f"{q['best_ideal_expected_cut']/22:.6f}", q["optimizer"]["total_evaluations"]])
    report += md_table(["p", "Gammas (radians)", "Betas (radians)", "Exact ideal mean", "Mean ratio", "Objective evaluations"], optimizer_rows)
    report += f'''

![p=1 convergence](results/figures/qaoa_convergence_p1.png)
![p=2 convergence](results/figures/qaoa_convergence_p2.png)

## 3. Architecture design and controlled comparison

{topology_table}

A is the five-node chain. B's core links are `(0,3),(0,4),(1,2),(1,3),(2,3),(2,4)`, the six logical pair interactions. Its remaining six links connect the four additional physical qubits into a sparse connected extension. Three original graph edges touching fixed node 0 become single-qubit phases, so they require no coupling links. B is a specific algorithm/architecture co-design; the four extra nodes are idle for this primary identity-layout instance, and the connectivity of the active core accounts for its routing advantage.

![Processor A](results/figures/architecture_processor_A.png)
![Processor B](results/figures/architecture_processor_B.png)

The same optimized angles, logical gate order, identity initial placement, optimization level 1, transpiler seed 42, and native basis `rz,sx,x,cx` are used on both processors. Initial placement is declared explicitly; an adaptive layout search might reduce A's routing burden, so these numbers are conditional on that policy. Tests additionally check a non-identity placement on both processors.

SWAPs are counted by an analysis-only `post_routing` hook before translation in the same preset pass manager that generates the simulated circuit. A separate target basis is never used to infer this count. Final basis circuits contain zero literal SWAPs because translation decomposes them. Extra CX count relative to an unconstrained compile is reported separately, and is not assumed to be a SWAP count. Final optimization can cancel CX gates, so the two measures need not scale as exactly three CX per SWAP.

Every final CX is checked against the coupling graph. The five measured physical qubits are reordered by destination classical bit after routing; the ideal output probabilities are then compared with the logical reference. The largest recorded deviation is below 1e-10. Gate count excludes measurement/barriers, while reported circuit depth includes the final measurements; unitary depth is also saved.

## 4. Noise, metrics and statistical interpretation

Both processors use one-qubit depolarizing parameter 0.001 on `rz,sx,x`, CX depolarizing parameter 0.005, and independent symmetric readout flip probability 0.01. These are the guide's synthetic rates, rather than measured hardware error rates. RZ receives depolarizing noise even though some physical platforms implement it virtually; this choice is held identical on A/B.

Exact ideal statevectors establish compiler equivalence. Exact noisy density matrices establish the effect of gate noise; applying the independent readout confusion channel to the saved pre-measurement probabilities incorporates readout error once. Full 32-outcome ideal/noisy probabilities are exported. Separate Aer evaluations sample 4096 shots each, for five seeds (42-46), for both noise modes and depths. The official primary comparison uses seed 42; repetitions quantify sampling sensitivity.

The mean approximation ratio is `E[cut]/22`. The best-observed ratio is also saved, but reaching 22 in one sample does not mean the mean ratio is 1. Optimal-cut probability is the total probability of all exact-optimum outcomes. Mean-cut uncertainty uses sample standard deviation / sqrt(shots), with approximate 95% normal intervals; optimal-cut rate uses Wilson 95% intervals.

The classical distribution fidelity is `(sum sqrt(p_logical*p_output))^2`, in [0,1], and total variation is `sum abs(p_logical-p_output)/2`. These describe measurement distributions, not quantum state fidelity or a proof of computational advantage.

## 5. Resource counts and exact modeled noise results

{summary_table}

'''
    for p in (1, 2):
        a, b = tables[p].iloc[0], tables[p].iloc[1]
        report += f"At p={p}, B reduces depth by **{100*(a.circuit_depth-b.circuit_depth)/a.circuit_depth:.1f}%** and CX count by **{100*(a.cx_count-b.cx_count)/a.cx_count:.1f}%**. A inserts {int(a.routing_swap_count)} SWAPs and B inserts {int(b.routing_swap_count)}. Exact noisy expected cut is higher on B by **{b.exact_noisy_expected_cut-a.exact_noisy_expected_cut:.6f}**. The exact result removes sampling uncertainty from that comparison, within the stated model and compilation policy.\n\n"
    report += "![p=1 resources](results/figures/processor_comparison_p1.png)\n![p=2 resources](results/figures/processor_comparison_p2.png)\n\n## 6. Shot-based results and repetition check\n\n"
    shot_rows, repeat_rows = [], []
    for p in (1, 2):
        for row in tables[p].itertuples():
            shot_rows.append([p, row.processor, f"{row.ideal_expected_cut_from_shots:.4f}", f"{row.noisy_expected_cut_from_shots:.4f}",
                              f"[{row.noisy_cut_mean_ci95_low:.4f}, {row.noisy_cut_mean_ci95_high:.4f}]", f"{row.noisy_approximation_ratio_of_sample_mean:.4f}",
                              f"{100*row.noisy_optimal_cut_shot_rate:.2f}%", row.noisy_best_observed_cut])
        repetitions = pd.read_csv(ROOT / f"results/tables/sampling_repetitions_p{p}.csv")
        for processor in ("Processor A", "Processor B"):
            noisy = repetitions[(repetitions.processor == processor) & (repetitions['mode'] == 'noisy')]
            repeat_rows.append([p, processor, len(noisy), int(noisy.shots.sum()), f"{noisy.expected_cut_from_samples.mean():.4f}",
                                f"{noisy.expected_cut_from_samples.std(ddof=1):.4f}", f"{100*noisy.optimal_cut_shot_rate.mean():.2f}%"])
    report += md_table(["p", "Processor", "Ideal shot mean", "Noisy shot mean", "Noisy mean 95% CI", "Mean ratio", "Optimum shot rate", "Best cut"], shot_rows)
    report += "\n\nFive repetitions, 4096 shots per evaluation:\n\n" + md_table(["p", "Processor", "Runs", "Total noisy shots", "Mean of run means", "SD of run means", "Mean optimum rate"], repeat_rows)
    report += '''

The noisy samples are consistent with the exact noisy predictions. Small ideal sample differences between A and B come from finite-shot sampling; exact ideal distributions agree. Noise can redistribute probability toward particular cuts, so an optimal-hit rate need not decline monotonically for arbitrary parameter choices. Here, expected-cut degradation and distribution fidelity provide broader indicators.

![p=1 shot distributions](results/figures/noisy_cut_distribution_p1.png)
![p=2 shot distributions](results/figures/noisy_cut_distribution_p2.png)
![Best sampled solution](results/figures/classical_vs_qaoa_cut_p2.png)

## 7. Depth trade-off, interpretation and limitations

p=2 increases the exact ideal expected cut and the exact noisy expected cut for both architectures compared with p=1. It also nearly doubles the pair-interaction gate budget and increases noise accumulation. On this instance the improvement in variational expressivity outweighs that additional noise cost at the stated rates; deeper circuits beyond p=2 were not studied.

The experiment supports the conclusion that B's active core removes routing overhead for this reduced graph, lowering native depth and CX count and improving the exact noisy objective. This is a controlled simulation with uniform gate noise and fixed layout, rather than a universal ranking of processor quality.

- Only one generated topology and weight assignment is benchmarked. B was explicitly tailored to its interactions, and its four spare qubits do not participate in the primary circuit.
- Initial layout and seeded compiler heuristics affect overhead. Better layout or alternative gate scheduling could change A's results.
- The symmetry reduction preserves the optimization objective but changes the finite-depth ansatz compared with a six-qubit mixer.
- Grid/multistart COBYLA avoids several poor local solutions but provides no certificate of the best possible QAOA angles.
- Synthetic independent depolarizing/readout channels omit relaxation, crosstalk, timing and device-dependent errors. No real quantum processor was used.
- Exact classical enumeration is trivial for this graph. Sampling an optimal cut does not establish quantum speedup or scalability.
- Confidence intervals describe shot variation only, not model bias or robustness to new graph instances.

## 8. Reproducibility and evidence

The notebook executes the entire experiment in a fresh Python kernel. The runner exports weighted inputs, every reduced classical cut, Hamiltonian terms, optimizer traces and restarts, optimized angles, QPY circuit files, exact output probabilities, all sampled histograms, processor summaries, figures and environment manifests. Unit tests check the objective and circuit with independent calculations. The submission validator recomputes metrics and verifies source/artifact SHA256 hashes to reject stale results.

Primary command: `python run_experiment.py` (both depths; 4096 shots per evaluation; seeds 42-46; 16 optimization starts; up to 800 evaluations/start). Run `python -m pytest -q`, `python execute_notebook.py`, and `python validate_submission.py` for the complete local validation. The source zip had no Git metadata; the public upstream inspected before finalization was commit `8c1dcb92612e9fab14b2b42fedc09b8033a84ea2`. Source content hashes identify the final local run independently of its eventual Git commit.

'''
    report += md_table(["Package", "Validated version"], [["Python", manifest["python"]]] + list(manifest["package_versions"].items()))
    report += '''

## 9. Master-guide requirement mapping

| Guide requirement | Evidence | Local state |
| --- | --- | --- |
| One selected PS; team of 2-6 | README and submission.json; PS-O1; two named members | Complete |
| Six-node graph, p=1 or p=2, classical reference | Both allowed depths; saved inputs and full/reduced enumeration | Complete |
| Participant-defined A(5q)/B(9q), valid basis and common baseline | Processor JSON, routing checks, saved QPY | Complete |
| Depth, CX, SWAP overhead and solution accuracy | Comparison CSVs; exact/noisy objective, ratios, success and fidelity | Complete |
| Required modular structure, including routing.py | Root notebook and src modules | Complete |
| Pinned dependencies and reproducible notebook | Direct/full locks; fresh-kernel execution | Complete after execution audit |
| Architectural analysis and legible figures | This report; graph, coupling and benchmark plots | Complete |
| Final public GitHub content and portal submission | Repository URL and external portal acceptance | Requires upload/submission confirmation |

The deadline in the supplied guide is **October 14, 2026, 23:59 IST**. Local correctness and packaging cannot verify event registration or portal acceptance. `submission.json` intentionally records portal confirmation as false until the team completes that external step. No Phase 2 Processor C/D deliverables are required.

## References

- Supplied Qiskit Fall Fest 2026 Phase 1 Master Challenge Guide: pp. 1-3 (scope, benchmark, structure), p. 4 (PS-O1), p. 7 (rubric/checklist).
- [Qiskit transpiler and stage hooks](https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler).
- [Qiskit Aer simulator documentation](https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html).
'''
    (ROOT / "report.md").write_text(report, encoding="utf-8")

    # Show quality and model effect with separate exact predictions and shot estimates.
    for p, table in tables.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        x = np.arange(2)
        axes[0].bar(x - .18, table.exact_noisy_expected_cut, .35, label="Exact noisy prediction", color="#42A895")
        axes[0].bar(x + .18, table.noisy_expected_cut_from_shots, .35,
                    yerr=1.96 * table.noisy_cut_mean_standard_error, capsize=4, label="4096-shot mean (95% CI)", color="#4E79A7")
        axes[0].axhline(22, linestyle="--", color="#E6B655", label="Exact optimum")
        axes[0].axhline(13.5, linestyle=":", color="#7B8794", label="Uniform random mean")
        axes[0].set_ylabel("Cut value")
        axes[0].set_xticks(x, table.processor)
        axes[0].set_ylim(0, 25)
        axes[0].legend(fontsize=8)
        axes[1].bar(x, table.exact_noisy_optimal_cut_probability, color=["#E07A5F", "#42A895"])
        axes[1].set_xticks(x, table.processor)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Exact noisy probability of optimum")
        fig.suptitle(f"QAOA p={p}: solution quality under the common noise model")
        fig.tight_layout()
        fig.savefig(ROOT / f"results/figures/solution_quality_p{p}.png", dpi=180)
        plt.close(fig)

    cells = []
    def markdown(s): cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s): cells.append(nbformat.v4.new_code_cell(s))
    markdown("# QFF 2026 Phase 1 - PS-O1: Weighted Max-Cut\n\n**Inderpreet Kaur and Harshit Tripathi.** This notebook regenerates both allowed QAOA depths, all classical references, A/B compilations, exact and sampled noise results, and the saved figures. Run every cell in the project root using the pinned local environment.")
    code("from pathlib import Path\nimport importlib.metadata as metadata\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd()\nassert (ROOT / 'src/routing.py').is_file(), 'Run from the project root'\nfor package in ['qiskit', 'qiskit-aer', 'numpy', 'pandas', 'networkx', 'scipy']:\n    print(package, metadata.version(package))\nfrom src.problem import build_weighted_graph, build_cost_hamiltonian, cost_diagonal\nfrom run_experiment import initial_classical_stage, run_qaoa_stage")
    markdown("## Objective and five-qubit reduction\n\nThe guide specifies six graph nodes but Processor A has five physical qubits. Complement symmetry lets us fix node 0 to zero, preserving all objective values. The remaining five bits define the reduced QAOA ansatz; its finite-depth state need not equal a six-qubit mixer state. Topology seed and documented weight seed are both 42.")
    code("graph = build_weighted_graph()\noptimum, classical_assignment = initial_classical_stage(graph)\nprint('Classical optimum:', optimum)\ndisplay(pd.read_csv(ROOT / 'data/raw/maxcut_edges.csv'))\ndisplay(pd.read_csv(ROOT / 'results/tables/classical_all_cuts.csv', dtype={'node_bits_0_to_5': str}).head())\ndisplay(Image(filename=str(ROOT / 'results/figures/02_classical_optimum_cut.png')))")
    markdown("## Cost Hamiltonian\n\n$C=\\sum_{(u,v)}w_{uv}(I-Z_u Z_v)/2$, with $Z_0=+1$. Every Hamiltonian eigenvalue is checked against a classical cut. QAOA applies `RZ(-gamma*w)`, `RZZ(-gamma*w)`, then `RX(2*beta)`.")
    code("hamiltonian = build_cost_hamiltonian(graph)\nprint(hamiltonian)\nnp.testing.assert_allclose(hamiltonian.to_matrix(), np.diag(cost_diagonal(graph)), atol=1e-12)\nprint('All 32 Hamiltonian eigenvalues match the reduced objective.')")
    markdown("## Processor designs\n\nA is a five-qubit chain. B is a connected nine-qubit graph with a core tailored to the six free-variable interactions. Angles, initial layout, native basis, transpiler level/seed and noise rates are held fixed across the comparison. B's spare physical qubits are idle in the primary run.")
    code("for name in ['A', 'B']:\n    display(Image(filename=str(ROOT / f'results/figures/architecture_processor_{name}.png')))")
    markdown("## Run both allowed QAOA depths\n\nOptimize p=1 with grid-seeded 16-start COBYLA, then warm-start p=2 with the p=1 result and additional starts. Each architecture has 4096 shots per evaluation, five sampling seeds, ideal/noisy evaluation, and exact noisy density-matrix probabilities. This cell recreates the data and figures from scratch.")
    code("p1 = run_qaoa_stage(graph, optimum, classical_assignment, depth=1, shots=4096, seed=42, repeats=5, restarts=16, maxiter=800)\np2 = run_qaoa_stage(graph, optimum, classical_assignment, depth=2, shots=4096, seed=42, repeats=5, restarts=16, maxiter=800, warm_start=(p1['gammas'], p1['betas']))\nassert p2['best_ideal_expected_cut'] >= p1['best_ideal_expected_cut']\nprint('Exact ideal expected cuts:', p1['best_ideal_expected_cut'], p2['best_ideal_expected_cut'])")
    markdown("## Quantitative architecture and solution comparisons\n\nDepth includes measurement. SWAPs are captured after routing in the actual compilation pipeline. Classical distribution fidelity measures output probabilities, not quantum state fidelity. The mean approximation ratio is mean cut/22, even when the best sample reaches 22.")
    code("columns = ['processor', 'circuit_depth', 'cx_count', 'routing_swap_count', 'exact_logical_expected_cut', 'exact_noisy_expected_cut', 'exact_noisy_approximation_ratio', 'noisy_expected_cut_from_shots', 'noisy_cut_mean_standard_error', 'noisy_optimal_cut_shot_rate', 'noisy_best_observed_cut']\nfor depth in [1, 2]:\n    print(f'QAOA p={depth}')\n    display(pd.read_csv(ROOT / f'results/tables/processor_comparison_p{depth}.csv')[columns])\n    display(Image(filename=str(ROOT / f'results/figures/processor_comparison_p{depth}.png')))\n    display(Image(filename=str(ROOT / f'results/figures/noisy_cut_distribution_p{depth}.png')))")
    markdown("## Sampling repetitions and exact optimum\n\nEach of five repetitions still uses 4096 shots, as specified by the guide. Exact noisy predictions isolate the modeled noise effect; repetition variation describes measurement sampling. A sampled optimum is not evidence that every shot is optimal.")
    code("for depth in [1, 2]:\n    repeats = pd.read_csv(ROOT / f'results/tables/sampling_repetitions_p{depth}.csv')\n    display(repeats.groupby(['processor', 'mode']).agg(mean_cut=('expected_cut_from_samples', 'mean'), between_run_sd=('expected_cut_from_samples', 'std'), mean_optimal_rate=('optimal_cut_shot_rate', 'mean')))\ndisplay(Image(filename=str(ROOT / 'results/figures/classical_vs_qaoa_cut_p2.png')))")
    markdown("## Local guide audit and interpretation\n\nThe audit checks native coupling, full objective consistency, all saved histograms, source/artifact hashes and the official baseline. Both depths improve on uniform random partitions. B removes routing for this tailored instance; this does not imply universal superiority or quantum speedup. Full discussion and limitations are in `report.md`. Uploading the final repository and submitting its URL at the event portal remain external steps.")
    code("from validate_submission import audit\nchecks = audit(require_executed_notebook=False)\nprint('Local checks passed:', checks['checks_passed'])\nprint('Portal submission confirmed:', checks['external_requirements']['portal_submission_confirmed'])\ndisplay(Markdown(filename=str(ROOT / 'report.md')))")
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": manifest["python"]}})
    nbformat.write(notebook, ROOT / "main.ipynb")
    print("Built README.md, report.md, solution-quality figures and main.ipynb from measured results.")


if __name__ == "__main__":
    main()
