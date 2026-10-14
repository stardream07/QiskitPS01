# PS-O1: Weighted Max-Cut QAOA and processor co-design

**Qiskit Fall Fest 2026, Phase 1**  
**Authors:** Inderpreet Kaur and Harshit Tripathi.  
**Problem:** exactly PS-O1. All results below come from the included executable code.

## 1. Problem and classical reference

The objective is to maximize crossing edge weight on the six-node NetworkX graph generated with `n=6`, `p=0.5`, and topology seed 42. Weights use NumPy `default_rng(42)`, sampled as integers from 1 to 5 in sorted edge order; this seed is an explicit choice because the guide does not specify it. The nine edges have total weight 27, so uniform independent partitions have expected cut 13.5.

The exact optimum is **22**. A representative partition is node bits **001011**, i.e. group 0 = {0,1,3} and group 1 = {2,4,5}. Its crossing weights are 1+4+3+3+5+4+2=22. Both all-64 full assignments and all-32 reduced assignments are enumerated; exported tables retain every reduced cut.

![Weighted graph](results/figures/01_original_weighted_graph.png)
![Classical solution](results/figures/02_classical_optimum_cut.png)

## 2. Five-qubit formulation and QAOA

Complementing all six bits leaves every crossing edge unchanged. Fixing node 0 to 0 selects exactly one representative of each complementary pair and preserves the full set of attainable cut values. Logical `q0...q4` encode nodes 1...5. This objective-preserving reduction fits the five-qubit budget; its free-variable mixer defines a reduced ansatz, whose finite-p distribution need not equal an unreduced six-qubit QAOA distribution.

The diagonal cost Hamiltonian is

$$C=\sum_{(u,v)\in E}\frac{w_{uv}}{2}(I-Z_uZ_v),\quad Z_0=+1.$$

For an edge touching node 0 the interaction becomes a single Z. The stored `SparsePauliOp` retains the identity offset, so its diagonal matches classical cut values exactly. Starting from the uniform superposition, each QAOA layer uses `exp(-i gamma C)` and `exp(-i beta sum X)`. Omitting the irrelevant identity phase yields `RZ(-gamma*w)` and `RZZ(-gamma*w)`; the mixer is `RX(2*beta)`. A test explicitly checks the sign convention against diagonal phase evolution.

For p=1, a 65-by-41 gamma/beta grid across `[0,2*pi)` and `[0,pi)` supplies separated starting points. COBYLA refines 16 starts with up to 800 evaluations each and tolerance 1e-7. p=2 uses the optimized p=1 state as a feasible starting point, a split-layer start, and seeded random starts. Every objective evaluation is saved and the best evaluated angles are retained, including when a local run stops at its budget. The fast NumPy objective is independently verified against Qiskit Statevector. This search makes no global variational optimality claim.

| p | Gammas (radians) | Betas (radians) | Exact ideal mean | Mean ratio | Objective evaluations |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1715265 | 0.3907392 | 17.642993 | 0.801954 | 3772 |
| 2 | 0.1368719, 0.2917845 | 0.4514599, 0.2598094 | 19.240690 | 0.874577 | 4633 |

![p=1 convergence](results/figures/qaoa_convergence_p1.png)
![p=2 convergence](results/figures/qaoa_convergence_p2.png)

## 3. Architecture design and controlled comparison

| Processor | Physical qubits | Undirected edges | Diameter | Maximum degree | Density |
| --- | --- | --- | --- | --- | --- |
| Processor A | 5 | 4 | 4 | 2 | 0.400 |
| Processor B | 9 | 12 | 4 | 4 | 0.333 |

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

| p | Processor | Depth | CX | Routing SWAPs | Exact noisy mean cut | Mean ratio | P(optimum) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Processor A | 38 | 27 | 5 | 17.1530 | 0.7797 | 13.32% |
| 1 | Processor B | 21 | 12 | 0 | 17.3180 | 0.7872 | 13.88% |
| 2 | Processor A | 69 | 51 | 9 | 18.1763 | 0.8262 | 18.30% |
| 2 | Processor B | 38 | 24 | 0 | 18.5631 | 0.8438 | 19.84% |

At p=1, B reduces depth by **44.7%** and CX count by **55.6%**. A inserts 5 SWAPs and B inserts 0. Exact noisy expected cut is higher on B by **0.165051**. The exact result removes sampling uncertainty from that comparison, within the stated model and compilation policy.

At p=2, B reduces depth by **44.9%** and CX count by **52.9%**. A inserts 9 SWAPs and B inserts 0. Exact noisy expected cut is higher on B by **0.386808**. The exact result removes sampling uncertainty from that comparison, within the stated model and compilation policy.

![p=1 resources](results/figures/processor_comparison_p1.png)
![p=2 resources](results/figures/processor_comparison_p2.png)

## 6. Shot-based results and repetition check

| p | Processor | Ideal shot mean | Noisy shot mean | Noisy mean 95% CI | Mean ratio | Optimum shot rate | Best cut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Processor A | 17.6936 | 17.2183 | [17.0996, 17.3369] | 0.7826 | 13.87% | 22 |
| 1 | Processor B | 17.6670 | 17.3914 | [17.2767, 17.5060] | 0.7905 | 14.31% | 22 |
| 2 | Processor A | 19.2385 | 18.1768 | [18.0597, 18.2939] | 0.8262 | 19.07% | 22 |
| 2 | Processor B | 19.1948 | 18.5635 | [18.4567, 18.6702] | 0.8438 | 20.46% | 22 |

Five repetitions, 4096 shots per evaluation:

| p | Processor | Runs | Total noisy shots | Mean of run means | SD of run means | Mean optimum rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Processor A | 5 | 20480 | 17.1510 | 0.0415 | 13.49% |
| 1 | Processor B | 5 | 20480 | 17.3406 | 0.0302 | 14.08% |
| 2 | Processor A | 5 | 20480 | 18.1907 | 0.0493 | 18.51% |
| 2 | Processor B | 5 | 20480 | 18.5669 | 0.0345 | 20.12% |

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

| Package | Validated version |
| --- | --- |
| Python | 3.14.7 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| networkx | 3.6.1 |
| scipy | 1.17.0 |
| qiskit | 2.5.0 |
| qiskit-aer | 0.17.2 |
| nbformat | 5.11.1 |
| nbclient | 0.11.0 |
| ipykernel | 7.4.0 |
| pytest | 9.1.1 |

## References
- [Qiskit transpiler and stage hooks](https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler).
- [Qiskit Aer simulator documentation](https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html).
