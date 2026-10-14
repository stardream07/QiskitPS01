"""Problem definition and classical references for weighted Max-Cut.

The official guide specifies nx.erdos_renyi_graph(n=6, p=0.5, seed=42) and
integer edge weights in [1, 5]. The guide does not prescribe a weight RNG seed;
this project uses NumPy seed 42 so the generated instance is reproducible.

We exploit the symmetry of Max-Cut: complementing every bit leaves the cut
value unchanged. We can therefore fix x_0 = 0 and represent the six-node
problem with five decision bits, matching Processor A's five-qubit budget.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json

import networkx as nx
import numpy as np
import pandas as pd

N_NODES = 6
GRAPH_SEED = 42
WEIGHT_SEED = 42
EDGE_PROBABILITY = 0.5
MIN_WEIGHT = 1
MAX_WEIGHT = 5  # inclusive
N_LOGICAL_QUBITS = N_NODES - 1


def build_weighted_graph(graph_seed: int = GRAPH_SEED, weight_seed: int = WEIGHT_SEED) -> nx.Graph:
    """Build the reproducible six-node graph specified by the challenge guide."""
    graph = nx.erdos_renyi_graph(n=N_NODES, p=EDGE_PROBABILITY, seed=graph_seed)
    rng = np.random.default_rng(weight_seed)
    for u, v in sorted(graph.edges()):
        graph.edges[u, v]["weight"] = int(rng.integers(MIN_WEIGHT, MAX_WEIGHT + 1))
    # Keep nodes ordered 0..5, even if the generated graph is sparse.
    graph.add_nodes_from(range(N_NODES))
    return graph


def graph_edge_rows(graph: nx.Graph) -> list[dict]:
    return [
        {"u": int(u), "v": int(v), "weight": int(data["weight"])}
        for u, v, data in sorted(graph.edges(data=True))
    ]


def save_graph_data(graph: nx.Graph, root: Path, graph_seed: int = GRAPH_SEED, weight_seed: int = WEIGHT_SEED) -> None:
    """Persist generated graph inputs for reproducibility and inspection."""
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(graph_edge_rows(graph)).to_csv(raw_dir / "maxcut_edges.csv", index=False)
    record = {
        "problem_statement": "PS-O1: Max-Cut on Weighted Graphs",
        "num_nodes": N_NODES,
        "edge_probability": EDGE_PROBABILITY,
        "graph_seed": graph_seed,
        "weight_seed": weight_seed,
        "weight_range_inclusive": [MIN_WEIGHT, MAX_WEIGHT],
        "fixed_node": 0,
        "fixed_node_value": 0,
        "logical_qubits_after_symmetry_reduction": N_LOGICAL_QUBITS,
        "edges": graph_edge_rows(graph),
        "note": "Graph structure follows the guide's NetworkX recipe. Weight seed 42 is a documented project choice because the guide leaves the weight-generation seed unspecified."
    }
    (processed_dir / "maxcut_graph.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def full_assignment_from_reduced(bits: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Map five logical bits q0..q4 to graph nodes 1..5, with node 0 fixed to 0."""
    if len(bits) != N_LOGICAL_QUBITS:
        raise ValueError(f"Expected {N_LOGICAL_QUBITS} reduced bits, got {len(bits)}")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Decision bits must be binary")
    return (0, *(int(bit) for bit in bits))


def cut_value(graph: nx.Graph, assignment: tuple[int, ...] | list[int]) -> int:
    """Return total weight of edges whose endpoints are in different groups."""
    if len(assignment) != graph.number_of_nodes():
        raise ValueError("Assignment length must equal the number of graph nodes")
    if any(bit not in (0, 1) for bit in assignment):
        raise ValueError("Assignment bits must be binary")
    return int(sum(
        int(data["weight"])
        for u, v, data in graph.edges(data=True)
        if int(assignment[u]) != int(assignment[v])
    ))


def reduced_bits_from_integer(index: int, n_qubits: int = N_LOGICAL_QUBITS) -> tuple[int, ...]:
    """Qiskit-style little-endian decision bits: q0 is the least significant bit."""
    if not 0 <= index < (1 << n_qubits):
        raise ValueError("Basis index is outside the register")
    return tuple((index >> q) & 1 for q in range(n_qubits))


def enumerate_classical(graph: nx.Graph) -> tuple[pd.DataFrame, int, list[tuple[int, ...]]]:
    """Exhaustively enumerate 2**5 representatives; returns rows, optimum, optima."""
    rows: list[dict] = []
    best = -1
    best_assignments: list[tuple[int, ...]] = []
    for index in range(1 << N_LOGICAL_QUBITS):
        reduced = reduced_bits_from_integer(index)
        assignment = full_assignment_from_reduced(reduced)
        value = cut_value(graph, assignment)
        bitstring = "".join(str(bit) for bit in assignment)
        rows.append({
            "reduced_index": index,
            "node_bits_0_to_5": bitstring,
            "selected_group_A_nodes": ",".join(str(i) for i, b in enumerate(assignment) if b == 0),
            "selected_group_B_nodes": ",".join(str(i) for i, b in enumerate(assignment) if b == 1),
            "cut_value": value,
        })
        if value > best:
            best = value
            best_assignments = [assignment]
        elif value == best:
            best_assignments.append(assignment)
    table = pd.DataFrame(rows).sort_values(["cut_value", "node_bits_0_to_5"], ascending=[False, True]).reset_index(drop=True)
    return table, best, best_assignments



def enumerate_full_classical(graph: nx.Graph) -> tuple[int, list[tuple[int, ...]]]:
    """Enumerate all 2**6 assignments without symmetry reduction for verification."""
    best = -1
    best_assignments: list[tuple[int, ...]] = []
    for assignment in product((0, 1), repeat=graph.number_of_nodes()):
        value = cut_value(graph, assignment)
        if value > best:
            best = value
            best_assignments = [tuple(assignment)]
        elif value == best:
            best_assignments.append(tuple(assignment))
    return best, best_assignments


def assignment_to_qaoa_index_bits(assignment: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return q0..q4 bits (nodes 1..5) from a full six-node assignment."""
    return tuple(int(assignment[node]) for node in range(1, N_NODES))


def reduced_pair_edges(graph: nx.Graph) -> list[tuple[int, int, int]]:
    """Return graph edges between variable nodes as (logical_q_u, logical_q_v, weight).

    Original node i>=1 maps to logical qubit i-1. Edges touching fixed node 0
    become single-qubit Z terms in the QAOA cost phase.
    """
    output = []
    for u, v, data in sorted(graph.edges(data=True)):
        if u != 0 and v != 0:
            output.append((u - 1, v - 1, int(data["weight"])))
    return output


def cost_diagonal(graph: nx.Graph) -> np.ndarray:
    """The 32 diagonal eigenvalues of the five-qubit cut Hamiltonian."""
    return np.array([
        cut_value(graph, full_assignment_from_reduced(reduced_bits_from_integer(i)))
        for i in range(1 << N_LOGICAL_QUBITS)
    ], dtype=float)


def build_cost_hamiltonian(graph: nx.Graph):
    """Return C = sum w(1-Zu Zv)/2 with Z0 replaced by +1.

    Qiskit Pauli strings are big endian: logical q0 is the rightmost letter.
    The identity offset is retained for expectations and exact verification.
    """
    from qiskit.quantum_info import SparsePauliOp
    terms = [("I" * N_LOGICAL_QUBITS, sum(d["weight"] for _, _, d in graph.edges(data=True)) / 2)]
    for u, v, data in sorted(graph.edges(data=True)):
        label = ["I"] * N_LOGICAL_QUBITS
        for node in (u, v):
            if node != 0:
                label[N_LOGICAL_QUBITS - node] = "Z"
        terms.append(("".join(label), -float(data["weight"]) / 2))
    return SparsePauliOp.from_list(terms).simplify()
