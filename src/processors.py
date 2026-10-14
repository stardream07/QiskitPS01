"""Processor graph loaders, transpilation and noise-model utilities."""
from __future__ import annotations

import json
from pathlib import Path

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
    if any(not (0 <= int(u) < processor["num_qubits"] and 0 <= int(v) < processor["num_qubits"]) for u, v in processor["edges"]):
        raise ValueError(f"An edge in {path} uses a qubit index outside its declared budget")
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


def transpile_for_processor(circuit, processor: dict, seed: int = TRANSPILE_SEED):
    """Transpile a measured logical circuit to the official basis and coupling map."""
    from qiskit import transpile
    coupling_map = make_coupling_map(processor)
    # Explicitly map logical q0..q4 onto physical nodes 0..4. Processor B's
    # custom core is designed for the reduced Max-Cut interaction graph.
    layout = list(range(circuit.num_qubits))
    compiled = transpile(
        circuit,
        basis_gates=BASIS_GATES,
        coupling_map=coupling_map,
        initial_layout=layout,
        optimization_level=1,
        seed_transpiler=seed,
    )
    return compiled


def get_routing_stage_swap_count(circuit, processor: dict, seed: int = TRANSPILE_SEED) -> int:
    """Estimate inserted routing SWAPs before native-basis decomposition.

    This diagnostic intentionally retains the SWAP gate as an allowed target
    operation so inserted routing swaps can be counted. It is not the final
    hardware-basis circuit; CX count/depth must be taken from the final compile.
    """
    from qiskit import transpile
    coupling_map = make_coupling_map(processor)
    route_basis = ["h", "rx", "rz", "rzz", "cx", "swap"]
    routed = transpile(
        circuit,
        basis_gates=route_basis,
        coupling_map=coupling_map,
        initial_layout=list(range(circuit.num_qubits)),
        optimization_level=1,
        seed_transpiler=seed,
    )
    return int(routed.count_ops().get("swap", 0))
