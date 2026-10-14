"""Independent numerical checks of the objective, QAOA, and physical routing."""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from src.problem import (build_weighted_graph, cut_value, cost_diagonal, build_cost_hamiltonian,
                         enumerate_classical, enumerate_full_classical, full_assignment_from_reduced)
from src.qaoa import build_qaoa_circuit, qaoa_probabilities, optimize_qaoa
from src.processors import load_processor, make_noise_model, NOISE_RATES, BASIS_GATES
from src.routing import compile_for_processor, compiled_probabilities, validate_compiled_circuit
from src.metrics import measurement_key_to_assignment, summarize_counts, circuit_metrics
from src.evaluation import apply_readout_error, exact_noisy_probabilities

ROOT = Path(__file__).resolve().parents[1]


def test_official_graph_and_exact_reference():
    graph = build_weighted_graph()
    expected = {(0, 2): 1, (0, 3): 4, (0, 4): 4, (1, 4): 3, (1, 5): 3,
                (2, 3): 5, (2, 4): 1, (3, 4): 4, (3, 5): 2}
    assert {(u, v): d["weight"] for u, v, d in graph.edges(data=True)} == expected
    reduced, optimum, optima = enumerate_classical(graph)
    full_optimum, full_optima = enumerate_full_classical(graph)
    assert len(reduced) == 32 and optimum == full_optimum == 22
    assert (0, 0, 1, 0, 1, 1) in optima
    assert len(full_optima) == 2 * len(optima)
    for bits in product((0, 1), repeat=6):
        assert cut_value(graph, bits) == cut_value(graph, tuple(1 - b for b in bits))


def test_hamiltonian_diagonal_matches_all_classical_cuts():
    graph = build_weighted_graph()
    matrix = build_cost_hamiltonian(graph).to_matrix()
    np.testing.assert_allclose(matrix, np.diag(cost_diagonal(graph)), atol=1e-12)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("seed", [7, 42, 103])
def test_numpy_qaoa_matches_independent_qiskit_statevector(depth, seed):
    graph = build_weighted_graph()
    rng = np.random.default_rng(seed)
    gammas, betas = rng.uniform(-4, 4, depth), rng.uniform(-2, 2, depth)
    reference = Statevector.from_instruction(build_qaoa_circuit(graph, gammas, betas)).probabilities()
    np.testing.assert_allclose(qaoa_probabilities(cost_diagonal(graph), gammas, betas), reference, atol=1e-12)
    assert np.isclose(reference.sum(), 1)


def test_cost_rotation_sign_and_mixer_independently():
    graph = build_weighted_graph()
    gamma, beta = 0.397, 0.218
    initial = np.ones(32, dtype=complex) / np.sqrt(32)
    cost_state = np.exp(-1j * gamma * cost_diagonal(graph)) * initial
    mixer = QuantumCircuit(5)
    mixer.rx(2 * beta, range(5))
    exact = Statevector(cost_state).evolve(mixer).probabilities()
    actual = Statevector.from_instruction(build_qaoa_circuit(graph, [gamma], [beta])).probabilities()
    np.testing.assert_allclose(actual, exact, atol=1e-12)


@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("layout", [[0, 1, 2, 3, 4], [4, 2, 0, 3, 1]])
def test_physical_routing_preserves_distribution_and_native_constraints(name, depth, layout):
    graph = build_weighted_graph()
    processor = load_processor(ROOT / f"processors/processor_{name}.json")
    circuit = build_qaoa_circuit(graph, [.37, .81][:depth], [.21, .43][:depth], True)
    compiled, info = compile_for_processor(circuit, processor, layout=layout)
    reference = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities()
    np.testing.assert_allclose(compiled_probabilities(compiled), reference, atol=1e-12)
    validate_compiled_circuit(compiled, processor)
    # Analysis-only hook must produce the same circuit as ordinary transpile.
    ordinary = transpile(circuit, basis_gates=BASIS_GATES, coupling_map=__import__('src.processors', fromlist=['make_coupling_map']).make_coupling_map(processor),
                         initial_layout=layout, seed_transpiler=42, optimization_level=1)
    assert compiled == ordinary
    assert info["routing_swap_count"] >= 0
    if name == "B" and layout == [0, 1, 2, 3, 4]:
        assert info["routing_swap_count"] == 0


def test_measurement_endianness_and_statistics():
    graph = build_weighted_graph()
    assert measurement_key_to_assignment("11010") == (0, 0, 1, 0, 1, 1)
    summary, table = summarize_counts(graph, {"11010": 75, "00000": 25}, 22)
    assert summary["expected_cut_from_samples"] == 16.5
    assert summary["approximation_ratio_of_sample_mean"] == .75
    assert summary["optimal_cut_shot_rate"] == .75
    assert summary["cut_mean_ci95_low"] < 16.5 < summary["cut_mean_ci95_high"]
    assert summary["optimal_cut_rate_ci95_low"] < .75 < summary["optimal_cut_rate_ci95_high"]
    assert table["shots"].sum() == 100
    ref = np.zeros(32)
    ref[26], ref[0] = .75, .25
    measured, _ = summarize_counts(graph, {"11010": 75, "00000": 25}, 22, ref)
    assert measured["classical_distribution_fidelity"] == pytest.approx(1)
    assert measured["total_variation_distance"] == 0


@pytest.mark.parametrize("counts", [{}, {"00000": 0}, {"00000": -1}, {"00000": 1.5}, {"101": 3}, {"abcde": 3}])
def test_invalid_histograms_rejected(counts):
    with pytest.raises(ValueError):
        summarize_counts(build_weighted_graph(), counts, 22)


@pytest.mark.parametrize("change", [{"num_qubits": 6}, {"edges": [[0, 0]]}, {"edges": [[0, 5]]},
                                    {"edges": [[0, 1], [1, 0]]}, {"edges": [[0, 1]]}, {"benchmark": {}}])
def test_invalid_processors_rejected(tmp_path, change):
    config = json.loads((ROOT / "processors/processor_A.json").read_text())
    config.update(change)
    path = tmp_path / "processor.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_processor(path)


def test_noise_model_uses_identical_official_rates():
    config = make_noise_model().to_dict()
    errors = config["errors"]
    assert {tuple(e["operations"]) for e in errors} == {("rz",), ("sx",), ("x",), ("cx",), ("measure",)}
    readout = next(e for e in errors if e["type"] == "roerror")
    np.testing.assert_allclose(readout["probabilities"], [[.99, .01], [.01, .99]])
    from qiskit_aer.noise import depolarizing_error
    for n, ops, rate in [(1, ["rz"], .001), (1, ["sx"], .001), (1, ["x"], .001), (2, ["cx"], .005)]:
        actual = next(e for e in errors if e["operations"] == ops)
        np.testing.assert_allclose(actual["probabilities"], depolarizing_error(rate, n).to_dict()["probabilities"])


def test_sampling_smoke_with_routing_and_noise():
    graph = build_weighted_graph()
    processor = load_processor(ROOT / "processors/processor_A.json")
    compiled, _ = compile_for_processor(build_qaoa_circuit(graph, [.3], [.2], True), processor)
    counts = AerSimulator(method="density_matrix", noise_model=make_noise_model(), max_parallel_threads=1).run(compiled, shots=256, seed_simulator=42).result().get_counts()
    summary, _ = summarize_counts(graph, counts, 22)
    assert summary["shots"] == 256 and 0 <= summary["expected_cut_from_samples"] <= 22
    assert circuit_metrics(compiled)["measurement_count"] == 5


def test_optimizer_retains_warm_start_and_is_reproducible():
    graph = build_weighted_graph()
    warm = ([.397], [.218])
    baseline = float(qaoa_probabilities(cost_diagonal(graph), *warm) @ cost_diagonal(graph))
    first = optimize_qaoa(graph, p=2, restarts=2, maxiter=30, warm_start=warm)
    second = optimize_qaoa(graph, p=2, restarts=2, maxiter=30, warm_start=warm)
    assert first[2] >= baseline - 1e-12
    np.testing.assert_allclose(np.r_[first[0], first[1]], np.r_[second[0], second[1]], atol=1e-12)
    assert first[2] == max(first[3])


def test_exact_noisy_distribution_accounts_for_readout_once():
    qc = QuantumCircuit(5, 5)
    qc.measure(range(5), range(5))
    expected = np.array([.01 ** i.bit_count() * .99 ** (5 - i.bit_count()) for i in range(32)])
    np.testing.assert_allclose(exact_noisy_probabilities(qc), expected, atol=1e-12)
    ideal = np.zeros(32)
    ideal[0] = 1
    np.testing.assert_allclose(apply_readout_error(ideal), expected, atol=1e-12)


def test_density_matrix_probabilities_match_sampled_noisy_mean():
    graph = build_weighted_graph()
    processor = load_processor(ROOT / "processors/processor_A.json")
    qc, _ = compile_for_processor(build_qaoa_circuit(graph, [.3], [.2], True), processor)
    probabilities = exact_noisy_probabilities(qc)
    counts = AerSimulator(method="density_matrix", noise_model=make_noise_model(), max_parallel_threads=1).run(qc, shots=8192, seed_simulator=42).result().get_counts()
    summary, _ = summarize_counts(graph, counts, 22)
    assert abs(summary["expected_cut_from_samples"] - float(probabilities @ cost_diagonal(graph))) < 6 * summary["cut_mean_standard_error"]
