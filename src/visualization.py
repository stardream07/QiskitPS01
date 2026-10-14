"""2D visualizations for the graph, solution, architectures and metrics."""
from __future__ import annotations

from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#19263D"
BLUE = "#4E79A7"
TEAL = "#42A895"
CORAL = "#E07A5F"
GOLD = "#E6B655"
LAVENDER = "#9D8BC7"
GREY = "#7B8794"
LIGHT = "#EEF2F7"


def _save(fig, path: Path | None):
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def plot_weighted_graph(graph, path: Path, assignment=None, title="Six-node weighted Max-Cut graph"):
    pos = nx.spring_layout(graph, seed=42, weight="weight")
    fig, ax = plt.subplots(figsize=(8.3, 5.8))
    node_colors = [TEAL if assignment and assignment[n] == 0 else CORAL if assignment else BLUE for n in graph.nodes]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=1.5, edge_color=GREY, alpha=0.85)
    if assignment is not None:
        cut_edges = [(u, v) for u, v in graph.edges if assignment[u] != assignment[v]]
        nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, ax=ax, width=3.0, edge_color=GOLD)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=900, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_color="white", font_size=12, font_weight="bold")
    edge_labels = {(u, v): d["weight"] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_color=NAVY, font_size=10, rotate=False, label_pos=0.5, bbox={"alpha": 0.85, "color": "white", "pad": 1})
    ax.set_title(title, fontsize=15, color=NAVY, pad=14, fontweight="bold")
    ax.text(0.5, -0.04, "Gold edges cross the cut; node colours show the two groups.", transform=ax.transAxes, ha="center", color=GREY, fontsize=9)
    ax.axis("off")
    _save(fig, path)


def plot_processor_graph(processor: dict, path: Path):
    graph = nx.Graph()
    graph.add_nodes_from(range(int(processor["num_qubits"])))
    graph.add_edges_from([tuple(map(int, e)) for e in processor["edges"]])
    pos = nx.spring_layout(graph, seed=42, k=1.0 / np.sqrt(max(1, graph.number_of_nodes())))
    fig, ax = plt.subplots(figsize=(8.3, 5.7))
    used = set(range(min(5, int(processor["num_qubits"]))))
    colors = [TEAL if n in used else LAVENDER for n in graph.nodes]
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=GREY, width=1.8, alpha=0.9)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=700, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_color="white", font_size=10, font_weight="bold")
    ax.set_title(f"{processor['name']} — {processor['num_qubits']} physical qubits", fontsize=15, color=NAVY, pad=14, fontweight="bold")
    ax.text(0.5, -0.03, "Teal: designated logical-qubit region (physical nodes 0–4); lavender: extra physical qubits.", transform=ax.transAxes, ha="center", color=GREY, fontsize=8.5)
    ax.axis("off")
    _save(fig, path)


def plot_cut_comparison(graph, classical_assignment, qaoa_assignment, classical_value, qaoa_value, path: Path):
    pos = nx.spring_layout(graph, seed=42, weight="weight")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, assignment, title, value in [
        (axes[0], classical_assignment, "Exact classical optimum", classical_value),
        (axes[1], qaoa_assignment, "Best cut observed in QAOA samples", qaoa_value),
    ]:
        colors = [TEAL if assignment[n] == 0 else CORAL for n in graph.nodes]
        cut_edges = [(u, v) for u, v in graph.edges if assignment[u] != assignment[v]]
        nx.draw_networkx_edges(graph, pos, ax=ax, width=1.4, edge_color=GREY, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, ax=ax, width=3.0, edge_color=GOLD)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=650, edgecolors="white", linewidths=2)
        nx.draw_networkx_labels(graph, pos, ax=ax, font_color="white", font_size=11, font_weight="bold")
        labels = {(u, v): d["weight"] for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=ax, font_size=9, font_color=NAVY, bbox={"alpha": 0.8, "color": "white", "pad": 1})
        ax.set_title(f"{title}\nCut value = {value}", color=NAVY, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Classical vs quantum-search cut", fontsize=16, color=NAVY, fontweight="bold")
    _save(fig, path)


def plot_convergence(history: list[float], optimum: int, path: Path, p: int):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    values = np.asarray(history, dtype=float)
    if len(values):
        best_so_far = np.maximum.accumulate(values)
        ax.plot(np.arange(1, len(best_so_far) + 1), best_so_far, color=TEAL, linewidth=2.2, label="Best expected cut so far")
        ax.scatter([len(values)], [best_so_far[-1]], color=CORAL, s=45, zorder=4)
    ax.axhline(optimum, color=GOLD, linestyle="--", linewidth=1.8, label=f"Classical optimum = {optimum}")
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Expected cut value")
    ax.set_title(f"QAOA parameter optimization (p={p})", color=NAVY, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_benchmark_bars(comparison: pd.DataFrame, path: Path):
    metrics = [("circuit_depth", "Circuit depth"), ("cx_count", "CX gate count"), ("routing_swap_count", "Routing SWAP count (pre-decomposition)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9))
    color_map = {"Processor A": CORAL, "Processor B": TEAL}
    for ax, (metric, label) in zip(axes, metrics):
        vals = comparison[metric].to_numpy(dtype=float)
        labels = comparison["processor"].tolist()
        bars = ax.bar(labels, vals, color=[color_map.get(n, BLUE) for n in labels], width=0.58)
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
        ax.set_title(label, fontsize=11, color=NAVY, fontweight="bold")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Processor architecture comparison", fontsize=15, color=NAVY, fontweight="bold")
    _save(fig, path)


def plot_cut_distribution(count_tables: dict[str, pd.DataFrame], path: Path):
    """Plot the ideal and noisy shot distributions for both processor models.

    Dictionary keys should follow e.g. 'Processor A — ideal' and
    'Processor A — noisy'. Colour identifies the processor; line style shows
    whether the sample came from an ideal or noisy run.
    """
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    colors = {"Processor A": CORAL, "Processor B": TEAL}
    for name, table in count_tables.items():
        if table.empty:
            continue
        grouped = table.groupby("cut_value")["shots"].sum().sort_index()
        total = grouped.sum()
        processor_name = "Processor A" if name.startswith("Processor A") else "Processor B"
        linestyle = "--" if "ideal" in name.lower() else "-"
        marker = "s" if "ideal" in name.lower() else "o"
        ax.plot(grouped.index, grouped / total, marker=marker, linestyle=linestyle, linewidth=2.0,
                markersize=4.5, color=colors.get(processor_name, BLUE), label=name)
    ax.set_xlabel("Observed cut value")
    ax.set_ylabel("Shot frequency")
    ax.set_title("Ideal and noisy QAOA output distributions", color=NAVY, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    _save(fig, path)
