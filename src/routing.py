"""Single-pipeline native compilation, routing evidence, and layout checks."""
from __future__ import annotations

import numpy as np
from .processors import BASIS_GATES, TRANSPILE_SEED, make_coupling_map


def compile_for_processor(circuit, processor, seed=TRANSPILE_SEED, layout=None):
    """Count SWAPs after routing in the SAME pipeline used for simulation.

    An analysis-only post-routing hook records the DAG before translation.
    It does not change routing or native optimization. Initial identity layout
    is held fixed on A and B to make the architecture comparison explicit.
    """
    from qiskit.transpiler import AnalysisPass, PassManager
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    class CaptureRouting(AnalysisPass):
        def run(self, dag):
            self.property_set["phase1_routing_swaps"] = int(dag.count_ops().get("swap", 0))
            self.property_set["phase1_routing_depth"] = int(dag.depth())

    if circuit.num_qubits > processor["num_qubits"]:
        raise ValueError("Logical circuit exceeds the physical qubit budget")
    layout = list(range(circuit.num_qubits)) if layout is None else list(layout)
    if len(layout) != circuit.num_qubits or len(set(layout)) != len(layout) or any(type(q) is not int or not 0 <= q < processor["num_qubits"] for q in layout):
        raise ValueError("Initial layout must map each logical qubit to a distinct physical qubit")
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=BASIS_GATES,
                                      coupling_map=make_coupling_map(processor),
                                      initial_layout=layout, seed_transpiler=seed)
    pm.post_routing = PassManager([CaptureRouting()])
    compiled = pm.run(circuit)
    validate_compiled_circuit(compiled, processor)
    info = {"routing_swap_count": pm.property_set["phase1_routing_swaps"],
            "routing_depth_before_translation": pm.property_set["phase1_routing_depth"],
            "initial_layout": layout, "measurement_physical_qubits_c0_first": measurement_qubits(compiled),
            "swap_count_method": "analysis-only post_routing hook in the final compilation pipeline"}
    return compiled, info


def measurement_qubits(circuit):
    """Physical measurement qubits ordered by classical bit c0,c1,... ."""
    mapping = {}
    for item in circuit.data:
        if item.operation.name == "measure":
            c = circuit.find_bit(item.clbits[0]).index
            if c in mapping:
                raise ValueError("Repeated classical measurement destination")
            mapping[c] = circuit.find_bit(item.qubits[0]).index
    if set(mapping) != set(range(circuit.num_clbits)):
        raise ValueError("Every classical bit must have one measurement")
    return [mapping[c] for c in range(circuit.num_clbits)]


def compiled_probabilities(circuit):
    """Exact ideal distribution in logical measurement order after routing."""
    from qiskit.quantum_info import Statevector
    return np.asarray(Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
                      .probabilities(qargs=measurement_qubits(circuit)), dtype=float)


def validate_compiled_circuit(circuit, processor):
    if circuit.num_qubits != processor["num_qubits"]:
        raise AssertionError("Compiled circuit does not use the declared physical register size")
    links = {tuple(sorted(edge)) for edge in processor["edges"]}
    allowed = set(BASIS_GATES) | {"measure", "barrier"}
    for item in circuit.data:
        name = item.operation.name
        if name not in allowed:
            raise AssertionError(f"Non-native instruction: {name}")
        if name == "cx":
            endpoints = tuple(sorted(circuit.find_bit(q).index for q in item.qubits))
            if endpoints not in links:
                raise AssertionError(f"CX violates hardware coupling: {endpoints}")
    measurement_qubits(circuit)
