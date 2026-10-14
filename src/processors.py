"""Processor graph loaders, transpilation and noise-model utilities."""
from __future__ import annotations

import json
from pathlib import Path
import networkx as nx

BASIS_GATES = ["rz", "sx", "x", "cx"]
TRANSPILE_SEED = 42
SHOTS = 4096
NOISE_RATES = {"single_qubit_depolarizing": 0.001, "cx_depolarizing": 0.005, "readout": 0.01}


def load_processor(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        processor = json.load(handle)
    required = {"name", "num_qubits", "edges"}
    missing = required - processor.keys()
    if missing:
        raise ValueError(f"Processor file {path} is missing required keys: {sorted(missing)}")
    n = processor["num_qubits"]
    if type(n) is not int or n not in (5, 9):
        raise ValueError("Phase 1 processor budgets must be 5 or 9 physical qubits")
    if not isinstance(processor["edges"], list) or not processor["edges"]:
        raise ValueError("A processor must specify a nonempty list of edges")
    unique = set()
    for edge in processor["edges"]:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("Each coupling edge must contain exactly two endpoints")
        u, v = edge
        if type(u) is not int or type(v) is not int or not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError("Coupling endpoints must be distinct integer qubits within the budget")
        link = tuple(sorted(edge))
        if link in unique:
            raise ValueError("Duplicate undirected coupling edge")
        unique.add(link)
    topology = nx.Graph()
    topology.add_nodes_from(range(n))
    topology.add_edges_from(unique)
    if not nx.is_connected(topology):
        raise ValueError("All declared physical qubits must form a connected graph")
    config = processor.get("benchmark", {})
    expected = {"basis_gates": BASIS_GATES, "optimization_level": 1,
                "seed_transpiler": TRANSPILE_SEED, "shots": SHOTS, "noise_rates": NOISE_RATES}
    if config != expected:
        raise ValueError("Processor benchmark config must match the official common baseline")
    return processor


def make_coupling_map(processor: dict):
    """Build a bidirectional coupling map from undirected hardware links."""
    try:
        from qiskit.transpiler import CouplingMap
    except ImportError as exc:
        raise ImportError("Qiskit is required for processor transpilation. Install requirements.txt") from exc
    directed = []
    for u, v in processor["edges"]:
        directed.extend([(int(u), int(v)), (int(v), int(u))])
    return CouplingMap(couplinglist=directed)


def make_noise_model():
    """Create the challenge's synthetic noise model for both processor models."""
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    except ImportError as exc:
        raise ImportError("Qiskit Aer is required for noisy simulation. Install requirements.txt") from exc
    noise = NoiseModel()
    one_qubit_error = depolarizing_error(NOISE_RATES["single_qubit_depolarizing"], 1)
    two_qubit_error = depolarizing_error(NOISE_RATES["cx_depolarizing"], 2)
    readout_error = ReadoutError([
        [1.0 - NOISE_RATES["readout"], NOISE_RATES["readout"]],
        [NOISE_RATES["readout"], 1.0 - NOISE_RATES["readout"]],
    ])
    noise.add_all_qubit_quantum_error(one_qubit_error, ["rz", "sx", "x"])
    noise.add_all_qubit_quantum_error(two_qubit_error, ["cx"])
    noise.add_all_qubit_readout_error(readout_error)
    return noise
