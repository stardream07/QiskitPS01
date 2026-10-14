# PS-O1 Report — QAOA for Weighted Max-Cut

> Replace every `[fill after running]` field with results from the actual experiment. Do not report expected or invented values as measured outcomes.

## 1. Problem statement

We solve the official Phase 1 PS-O1 weighted Max-Cut problem on a six-node Erdős–Rényi graph. The objective is to partition the vertices into two groups to maximize the sum of weights on crossing edges. The guide permits QAOA depth `p=1` or `p=2` and asks for cut value, approximation ratio, circuit depth, and SWAP-routing analysis.

## 2. Reproducible graph instance

- Topology generation: `networkx.erdos_renyi_graph(n=6, p=0.5, seed=42)`.
- Integer edge weights: `[1,5]`, generated using NumPy seed `42` as a documented project choice because the guide does not specify a weight seed.
- Number of vertices: 6.
- Number of edges: [fill from `data/raw/maxcut_edges.csv`].
- Original graph: `results/figures/01_original_weighted_graph.png`.

## 3. Symmetry reduction to five logical qubits

Complementing every vertex bit exchanges the group labels but leaves the set of crossing edges unchanged. Thus every cut has a complementary cut of equal value, and exactly one of the pair has node 0 set to 0. We fix node 0 to 0 and encode nodes 1–5 on five logical qubits. Exhaustive enumeration of 32 assignments verifies that the reduced problem preserves the full Max-Cut optimum.

For this generated instance, the exact classical maximum is **22** with representative bit string `001011` (nodes 0 through 5). This result must be regenerated if the graph or weights change.

## 4. QAOA method

The cost value is

\[
C(x)=\sum_{(u,v)\in E}w_{uv}\frac{1-Z_uZ_v}{2},
\]

with terms touching the fixed node 0 simplified to single-qubit Z terms. The QAOA ansatz starts in a uniform superposition on five qubits and alternates the cost unitary and an RX mixer. A classical COBYLA optimizer maximizes the exact statevector expected cut for parameter selection. The selected parameters are then held fixed for both processor comparisons.

- QAOA depth `p`: [fill: 1 or 2].
- Optimizer and iteration configuration: [fill from `data/processed/qaoa_parameters_p*.json`].
- Best ideal expected cut after optimization: [fill from the same JSON].

## 5. Processor designs

### Processor A

Five physical qubits in a linear chain, matching the guide's baseline reference topology.

### Processor B

Nine physical qubits and twelve undirected edges. The five-node core contains the six pairwise interactions in this fixed reduced instance; four additional qubits form a sparse extension. This topology is intentionally tailored to this graph and should not be claimed as universally superior. Compare coupling diagrams under `results/figures/architecture_processor_*.png`.

## 6. Evaluation settings

- Basis gates: `rz`, `sx`, `x`, `cx`.
- Transpiler optimization level: 1.
- Transpiler seed: 42.
- Shots per ideal/noisy processor run: [fill; default 4096].
- Single-qubit depolarizing probability: 0.001.
- CX depolarizing probability: 0.005.
- Readout error probability: 0.01.
- Both architectures use identical synthetic noise rates.

## 7. Results

Insert `results/tables/processor_comparison_p*.csv` here. Discuss at minimum:

- Final circuit depth and CX count for each processor.
- Routing-stage SWAP count before native-basis decomposition.
- CX-count change versus the unconstrained compilation baseline (a proxy; not an exact SWAP count).
- Ideal sample-mean cut and optimal-cut shot rate.
- Noisy sample-mean cut, best observed cut, and optimal-cut shot rate.
- Approximation ratio and its definition used in this report.

Figures to include:

1. Original weighted graph.
2. Exact classical optimum partition.
3. Processor A and B coupling maps.
4. QAOA optimization trace.
5. Processor metric comparison.
6. Ideal/noisy observed cut distributions and classical-versus-QAOA partition.

## 8. Interpretation and limitations

Discuss whether the architecture with lower routing requirements also has lower depth/CX count, and whether that correlates with noisy results. Do not claim connectivity alone caused a measured difference unless the experiment supports it; layout, transpiler heuristics, finite shots, and the chosen noise model also matter.

Limitations: the graph/weights are one generated instance; Processor B is custom-designed to match this reduced instance; the synthetic noise model is simplified; and QAOA with low p is not guaranteed to find the exact optimum. The exact classical optimum is feasible here because there are only 32 symmetry-reduced assignments.

## 9. Reproducibility

Record Python, Qiskit, Qiskit Aer, NetworkX, NumPy, and SciPy versions. Include the final command used, repository commit, random seeds, and any changes made to the processor JSON files or QAOA settings.
