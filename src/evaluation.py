"""Exact noisy measurement distributions, including independent readout flips."""
import numpy as np
from .processors import NOISE_RATES, make_noise_model
from .routing import measurement_qubits


def apply_readout_error(probabilities, rate=NOISE_RATES["readout"]):
    """Apply the symmetric independent readout channel on each measured bit."""
    probabilities = np.array(probabilities, dtype=float, copy=True)
    n = int(np.log2(len(probabilities)))
    if len(probabilities) != 1 << n or not 0 <= rate <= 1:
        raise ValueError("Readout requires a power-of-two distribution and a probability rate")
    for q in range(n):
        pairs = probabilities.reshape(-1, 2, 1 << q)
        a, b = pairs[:, 0, :].copy(), pairs[:, 1, :].copy()
        pairs[:, 0, :], pairs[:, 1, :] = (1 - rate) * a + rate * b, rate * a + (1 - rate) * b
    return probabilities


def exact_noisy_probabilities(compiled):
    """Density-matrix probabilities plus the guide's readout confusion matrix.

    Readout noise acts on measurements, so a pre-measurement save does not
    include it. Apply the independent 1% bit flips explicitly exactly once.
    """
    from qiskit_aer import AerSimulator
    circuit = compiled.remove_final_measurements(inplace=False)
    circuit.save_probabilities(qubits=measurement_qubits(compiled), label="logical_probabilities")
    backend = AerSimulator(method="density_matrix", noise_model=make_noise_model(), max_parallel_threads=1)
    probabilities = backend.run(circuit, shots=1).result().data()["logical_probabilities"]
    result = apply_readout_error(probabilities)
    if np.min(result) < -1e-12 or not np.isclose(result.sum(), 1, atol=1e-10):
        raise AssertionError("Noisy distribution is not normalized")
    return np.clip(result, 0, 1)
