"""QAOA circuit construction and exact statevector parameter optimization."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .problem import (
    N_LOGICAL_QUBITS,
    cut_value,
    full_assignment_from_reduced,
    reduced_bits_from_integer,
)


def build_qaoa_circuit(graph, gammas, betas, measure: bool = False):
    """Build a QAOA Max-Cut circuit using five logical qubits.

    Node 0 is fixed to x0=0 to remove the global group-swap symmetry.
    Cost term for an edge (u,v) is w*(1 - Z_u Z_v)/2. For edges that touch
    fixed node 0, the term reduces to w*(1-Z_v)/2, hence a single RZ rotation
    is sufficient. Global phases are omitted because they do not affect results.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError("Qiskit is required for QAOA. Install dependencies with: python -m pip install -r requirements.txt") from exc

    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length (QAOA depth p)")
    p = len(gammas)
    qc = QuantumCircuit(N_LOGICAL_QUBITS, N_LOGICAL_QUBITS if measure else 0)
    for q in range(N_LOGICAL_QUBITS):
        qc.h(q)

    for gamma, beta in zip(gammas, betas):
        for u, v, data in sorted(graph.edges(data=True)):
            weight = int(data["weight"])
            if u == 0:
                # z_0 = +1 because node 0 is fixed to bit 0.
                qc.rz(-float(gamma) * weight, v - 1)
            elif v == 0:
                qc.rz(-float(gamma) * weight, u - 1)
            else:
                qc.rzz(-float(gamma) * weight, u - 1, v - 1)
        for q in range(N_LOGICAL_QUBITS):
            qc.rx(2.0 * float(beta), q)

    if measure:
        qc.measure(range(N_LOGICAL_QUBITS), range(N_LOGICAL_QUBITS))
    return qc


def expected_cut_statevector(graph, gammas, betas) -> float:
    """Exact expected Max-Cut value for a QAOA circuit using Statevector."""
    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise ImportError("Qiskit is required for statevector optimization. Install requirements.txt") from exc
    circuit = build_qaoa_circuit(graph, gammas, betas, measure=False)
    probabilities = Statevector.from_instruction(circuit).probabilities()
    expectation = 0.0
    for index, probability in enumerate(probabilities):
        reduced = reduced_bits_from_integer(index)
        assignment = full_assignment_from_reduced(reduced)
        expectation += float(probability) * cut_value(graph, assignment)
    return float(expectation)


def optimize_qaoa(graph, p: int = 1, seed: int = 42, restarts: int = 3, maxiter: int | None = None):
    """Optimize QAOA angles by minimizing negative exact expected cut value.

    Optimizer evaluation uses exact statevector probabilities for deterministic
    convergence. Processor-level ideal/noisy shot-based simulations happen later.
    Returns (gammas, betas, expected_cut, history, optimizer_summary).
    """
    if p not in (1, 2):
        raise ValueError("The official PS-O1 guide specifies QAOA p=1 or p=2 for this task.")
    if maxiter is None:
        maxiter = 100 if p == 1 else 160
    rng = np.random.default_rng(seed)
    best_result = None
    best_history = None

    for restart in range(restarts):
        initial = np.concatenate([
            rng.uniform(0.0, np.pi, size=p),
            rng.uniform(0.0, np.pi / 2.0, size=p),
        ])
        history: list[float] = []

        def objective(theta):
            gammas = theta[:p]
            betas = theta[p:]
            score = expected_cut_statevector(graph, gammas, betas)
            history.append(score)
            return -score

        result = minimize(
            objective,
            initial,
            method="COBYLA",
            options={"maxiter": int(maxiter), "rhobeg": 0.5, "tol": 1e-4},
        )
        score = -float(result.fun)
        if best_result is None or score > -float(best_result.fun):
            best_result = result
            best_history = history.copy()

    assert best_result is not None and best_history is not None
    return (
        np.asarray(best_result.x[:p], dtype=float),
        np.asarray(best_result.x[p:], dtype=float),
        -float(best_result.fun),
        best_history,
        {
            "success": bool(best_result.success),
            "message": str(best_result.message),
            "nfev": int(best_result.nfev),
            "nit": int(getattr(best_result, "nit", best_result.nfev)),
            "restarts": int(restarts),
            "maxiter_per_restart": int(maxiter),
        },
    )
