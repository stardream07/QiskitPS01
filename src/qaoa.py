"""QAOA circuits and reproducible multistart exact-expectation optimization."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from .problem import N_LOGICAL_QUBITS, cost_diagonal


def _angles(gammas, betas):
    gammas, betas = np.asarray(gammas, dtype=float), np.asarray(betas, dtype=float)
    if gammas.ndim != 1 or betas.ndim != 1 or len(gammas) != len(betas) or len(gammas) not in (1, 2):
        raise ValueError("QAOA requires equal one-dimensional gamma/beta arrays of depth 1 or 2")
    if not (np.isfinite(gammas).all() and np.isfinite(betas).all()):
        raise ValueError("QAOA angles must be finite")
    return gammas, betas


def build_qaoa_circuit(graph, gammas, betas, measure: bool = False):
    """Implement exp(-i gamma C) followed by exp(-i beta sum X).

    The identity phase in C is omitted. Node 0 is fixed at bit 0, so its
    incident edges become RZ(-gamma*w); other edges use RZZ(-gamma*w).
    This is QAOA on the reduced objective, with a mixer on the five free bits.
    """
    from qiskit import QuantumCircuit
    gammas, betas = _angles(gammas, betas)
    qc = QuantumCircuit(N_LOGICAL_QUBITS, N_LOGICAL_QUBITS if measure else 0)
    qc.h(range(N_LOGICAL_QUBITS))
    for gamma, beta in zip(gammas, betas):
        for u, v, data in sorted(graph.edges(data=True)):
            angle = -float(gamma) * int(data["weight"])
            if u == 0 or v == 0:
                qc.rz(angle, (v if u == 0 else u) - 1)
            else:
                qc.rzz(angle, u - 1, v - 1)
        qc.rx(2 * float(beta), range(N_LOGICAL_QUBITS))
    if measure:
        qc.measure(range(N_LOGICAL_QUBITS), range(N_LOGICAL_QUBITS))
    return qc


def qaoa_probabilities(diagonal, gammas, betas):
    """Small exact NumPy simulator, independently checked against Qiskit.

    Cost phases are diagonal. RX mixers update amplitude pairs that differ
    at one bit. This avoids rebuilding a Qiskit circuit on every evaluation.
    """
    gammas, betas = _angles(gammas, betas)
    diagonal = np.asarray(diagonal, dtype=float)
    if diagonal.shape != (1 << N_LOGICAL_QUBITS,):
        raise ValueError("Expected the 32-entry reduced cost diagonal")
    state = np.full(len(diagonal), 1 / np.sqrt(len(diagonal)), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * diagonal)
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(N_LOGICAL_QUBITS):
            pairs = state.reshape(-1, 2, 1 << q)
            a, b = pairs[:, 0, :].copy(), pairs[:, 1, :].copy()
            pairs[:, 0, :], pairs[:, 1, :] = c * a + s * b, s * a + c * b
    return np.abs(state) ** 2


def expected_cut_statevector(graph, gammas, betas) -> float:
    """Independent Qiskit exact statevector expectation."""
    from qiskit.quantum_info import Statevector
    probabilities = Statevector.from_instruction(build_qaoa_circuit(graph, gammas, betas)).probabilities()
    return float(probabilities @ cost_diagonal(graph))


def optimize_qaoa(graph, p=1, seed=42, restarts=16, maxiter=800, warm_start=None):
    """Grid-seeded COBYLA for p=1, warm-start plus random multistart for p=2.

    Every evaluation is recorded, and the best evaluated angles are retained
    even if a local run exhausts its budget. No global optimality is claimed.
    warm_start is the p=1 (gammas, betas) pair. Appending a zero second layer
    makes the p=1 state a feasible p=2 starting point.
    """
    if p not in (1, 2) or restarts < 1 or maxiter < 1:
        raise ValueError("Depth must be 1 or 2 and optimization budgets must be positive")
    diagonal = cost_diagonal(graph)
    rng = np.random.default_rng(seed)
    history, records = [], []
    best_score, best_theta = -np.inf, None

    def score(theta):
        nonlocal best_score, best_theta
        value = float(qaoa_probabilities(diagonal, theta[:p], theta[p:]) @ diagonal)
        history.append(value)
        if value > best_score:
            best_score, best_theta = value, np.asarray(theta).copy()
        return -value

    starts = []
    grid_evaluations = 0
    if p == 1:
        candidates = []
        # Integer weights give gamma period 2*pi; RX probabilities have beta period pi.
        for gamma in np.linspace(0, 2 * np.pi, 65, endpoint=False):
            for beta in np.linspace(0, np.pi, 41, endpoint=False):
                theta = np.array([gamma, beta])
                candidates.append((-score(theta), theta))
        grid_evaluations = len(candidates)
        # Spatial separation avoids spending every restart in one grid basin.
        for _, theta in sorted(candidates, key=lambda item: -item[0]):
            if all(np.linalg.norm(theta - other) > 0.25 for other in starts):
                starts.append(theta)
            if len(starts) == restarts:
                break
    elif warm_start is not None:
        g, b = _angles(*warm_start)
        if len(g) != 1:
            raise ValueError("A p=2 warm start must be a p=1 solution")
        starts.extend([np.array([g[0], 0, b[0], 0]), np.array([g[0] / 2, g[0] / 2, b[0] / 2, b[0] / 2])])
    while len(starts) < restarts:
        starts.append(np.r_[rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
    for restart, initial in enumerate(starts[:restarts]):
        score(initial)
        result = minimize(score, initial, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.35, "tol": 1e-7})
        records.append({"restart": restart, "initial_angles": initial.tolist(), "final_angles": result.x.tolist(),
                        "expected_cut": -float(result.fun), "success": bool(result.success),
                        "nfev": int(result.nfev), "message": str(result.message)})
    assert best_theta is not None
    checked = expected_cut_statevector(graph, best_theta[:p], best_theta[p:])
    if not np.isclose(checked, best_score, atol=1e-10):
        raise AssertionError("NumPy and Qiskit QAOA expectations disagree")
    info = {"method": "COBYLA", "restarts": restarts, "maxiter_per_restart": maxiter,
            "total_evaluations": len(history), "grid_evaluations": grid_evaluations,
            "tolerance": 1e-7, "warm_start_used": warm_start is not None,
            "qiskit_expected_cut_check": checked, "runs": records,
            "global_optimality_claimed": False}
    return best_theta[:p], best_theta[p:], best_score, history, info
