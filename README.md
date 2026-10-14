# QFF 2026 Phase 1 — PS-O1: Max-Cut on Weighted Graphs

**Goal:** implement QAOA for the official six-node weighted Max-Cut problem, then compare the same logical circuit on two participant-defined processor architectures.

This repository is a reproducible starter implementation. It separates the classical reference, QAOA construction, processor/transpilation logic, measurements, and plotting so the experiment is easy to inspect and explain.

## Why this problem has useful visualisations

Max-Cut is naturally visual: nodes are the problem variables, edge labels are weights, and the cut is shown by two node colours and highlighted crossing edges. The project produces 2D plots for the original graph, exact classical cut, processor coupling maps, QAOA convergence, processor metrics, output distributions, and a classical-versus-QAOA cut comparison. **3D visuals are not required.**

## Important modelling decisions

1. **Graph structure:** follows the guide's `networkx.erdos_renyi_graph(n=6, p=0.5, seed=42)` recipe.
2. **Edge weights:** the guide specifies integer weights from 1 to 5 but does not specify a random seed for those weights. This project documents NumPy seed `42` as a reproducibility choice. The actual generated graph and weights are saved in `data/raw/maxcut_edges.csv` and `data/processed/maxcut_graph.json`.
3. **Five-qubit reduction:** Max-Cut is invariant under swapping group labels. Complementing all six bits does not change which edges cross the cut. We therefore fix graph node 0 to bit 0, leaving five decision bits/qubits. This preserves the optimal cut value and lets the problem fit Processor A's five-qubit budget. The reduction is verified by enumerating all 32 representatives.
4. **QAOA depth:** default `p=1`; the script also supports `p=2`, both allowed by the guide. Run `python run_experiment.py --depth 2` for the second experiment.
5. **Processor A:** five-qubit linear chain, the guide's baseline reference layout.
6. **Processor B:** participant-designed nine-qubit, twelve-edge graph. Its five-node core directly contains the six pair interactions required by this fixed reduced instance; four extra qubits are connected through six sparse extension edges. This is a deliberate, problem-aware architecture choice, not a mandated topology. Discuss this custom tailoring as a limitation and do not claim it generalises to every Max-Cut graph.
7. **Noise:** both processors use identical synthetic rates from the guide: single-qubit depolarizing `p=0.001`, CX depolarizing `p=0.005`, and readout error `p=0.01`.
8. **Fair comparison:** QAOA parameters are optimized once on the exact five-qubit statevector and then the same logical circuit is compiled and sampled on both processor models. This helps isolate how mapping/connectivity and noise affect the result.
9. **Routing overhead:** the script reports SWAPs from a routing-stage diagnostic before basis decomposition and also reports the final CX count and the change from an unconstrained compilation baseline. The CX delta is a proxy, not an exact SWAP count. The final compiled native-gate circuits use the guide's `{rz, sx, x, cx}` basis.

## Classical reference result for this starter's generated instance

For the documented graph seed `42` and weight seed `42`, exhaustive search reports an exact maximum cut value of **22**, with one representative assignment `001011` for nodes 0 through 5 (node 0 fixed to 0). This is a classical result for this specific reproducible instance, **not a quantum result**. If you change the graph or weight seed, the optimum can change.

## Setup (recommended: local Python 3.12)

A browser-only notebook kernel may not support Qiskit Aer or compiled packages. Use a local environment on your computer, preferably Python 3.12.

### Windows PowerShell

```powershell
# Open a terminal in the extracted PS_O1_MaxCut_QAOA_Starter folder
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.12` says Python 3.12 is not installed, install Python 3.12 first, then repeat these commands. In VS Code, open `main.ipynb`, select the `.venv` Python kernel, and run cells from top to bottom.

### Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the experiment

First generate/check the graph, exact optimum, and initial plots. This step does not require Qiskit:

```bash
python run_experiment.py --classical-only
```

Then run QAOA optimization, compilation, and ideal/noisy simulations on both processors:

```bash
python run_experiment.py
```

By default, this uses `p=1` and `4096` shots per processor per ideal/noisy run, following the guide's benchmark baseline. Optional commands:

```bash
python run_experiment.py --depth 2
python run_experiment.py --depth 1 --shots 8192
```

The full QAOA run needs Qiskit and Qiskit Aer installed. It has not been executed in an environment without those packages; do not treat any benchmark table as complete until you run it locally. The bundled classical-only plots are real outputs generated by the included code.

## Output files

- `data/raw/maxcut_edges.csv` — graph edges and weights.
- `data/processed/maxcut_graph.json` — seeds and complete problem instance.
- `data/processed/qaoa_parameters_p*.json` — parameters and optimizer metadata.
- `results/tables/classical_all_cuts.csv` — all 32 symmetry-reduced partitions.
- `results/tables/classical_optimum.csv` — exact classical optimum.
- `results/tables/processor_comparison_p*.csv` — final side-by-side metrics.
- `results/tables/*_ideal_counts_p*.csv` and `*_noisy_counts_p*.csv` — measurement histograms.
- `results/figures/` — 2D diagrams and benchmark plots.
- `report.md` — report structure and prompts for interpreting actual results.

## Suggested report questions

1. Does the QAOA output include the exact classical optimum among its sampled cuts?
2. What fraction of ideal and noisy shots hit an optimal cut?
3. How do circuit depth, final CX count and routing-stage SWAP count compare?
4. How does the noisy sample mean compare with the ideal sample mean and exact optimum?
5. What can you conclude about this **specific** architecture/instance, and what does the custom Processor B design prevent you from generalizing?

## Repository structure

```text
PS_O1_MaxCut_QAOA_Starter/
├── README.md
├── main.ipynb
├── requirements.txt
├── run_experiment.py
├── data/
│   ├── raw/
│   └── processed/
├── processors/
│   ├── processor_A.json
│   └── processor_B.json
├── src/
│   ├── problem.py
│   ├── qaoa.py
│   ├── processors.py
│   ├── metrics.py
│   └── visualization.py
├── results/
│   ├── tables/
│   └── figures/
└── report.md
```
