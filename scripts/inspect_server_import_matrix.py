#!/usr/bin/env python
"""Aggregate intra-server import counts to visualise package boundaries."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, Iterable, Tuple

from grimp import build_graph


SERVER_PREFIX = "napari_cuda.server."


def classify(module: str, mapping: Dict[str, str]) -> str:
    """Return the domain bucket for a module (defaults to 'misc')."""
    if not module.startswith(SERVER_PREFIX):
        return "external"
    suffix = module[len(SERVER_PREFIX) :]
    head = suffix.split(".", 1)[0]
    return mapping.get(head, head or "misc")


def collect_edges(
    *,
    graph_root: str,
    include_external: bool,
    mapping: Dict[str, str],
) -> Iterable[Tuple[str, str]]:
    graph = build_graph(graph_root, include_external_packages=include_external)
    for module in graph.modules:
        if not module.startswith(SERVER_PREFIX):
            continue
        src_bucket = classify(module, mapping)
        for imported in graph.find_modules_directly_imported_by(module):
            dst_bucket = classify(imported, mapping)
            yield src_bucket, dst_bucket


def build_matrix(edges: Iterable[Tuple[str, str]]) -> Dict[str, Counter]:
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for src, dst in edges:
        matrix[src][dst] += 1
    return matrix


def render(matrix: Dict[str, Counter]) -> None:
    headers = sorted({bucket for bucket in matrix} | {dst for counter in matrix.values() for dst in counter})
    header_line = " " * 15 + " ".join(f"{h:>10}" for h in headers)
    print(header_line)
    for src in headers:
        row = matrix.get(src, Counter())
        counts = " ".join(f"{row.get(dst, 0):10d}" for dst in headers)
        print(f"{src:>15} {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise imports between server subpackages."
    )
    parser.add_argument(
        "--root",
        default="napari_cuda",
        help="Top-level package to analyse (default: %(default)s).",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include external packages in the graph.",
    )
    args = parser.parse_args()

    # Normalise naming for doc + viewstate under new contract.
    mapping = {
        "viewstate": "scene",
        "tests": "tests",
        "docs": "docs",
    }

    matrix = build_matrix(
        collect_edges(
            graph_root=args.root,
            include_external=args.include_external,
            mapping=mapping,
        )
    )
    render(matrix)


if __name__ == "__main__":
    main()

