"""Minimal IDS grammar utilities.

This module provides a tiny subset of the Ideographic Description
Sequences (IDS) syntax used for describing the structure of Hanzi
characters.  Only a handful of composition operators and components are
included which is sufficient for unit tests and light experimentation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterator, List, Sequence

IDS_TOKENS = {"⿰", "⿱", "⿴", "⿺", "⿻"}

# A very small component vocabulary.  The functions work with arbitrary
# symbols so this set can easily be extended by downstream users.
COMPONENTS = {"木", "口", "日", "人", "一"}


@dataclass
class IDSTree:
    """Simple tree structure representing an IDS expression."""

    value: str
    children: List["IDSTree"]


def _parse_it(it: Iterator[str]) -> IDSTree:
    try:
        ch = next(it)
    except StopIteration:  # pragma: no cover - defensive
        return IDSTree("", [])
    if ch in IDS_TOKENS:
        left = _parse_it(it)
        right = _parse_it(it)
        return IDSTree(ch, [left, right])
    return IDSTree(ch, [])


def parse_ids(s: str) -> IDSTree:
    """Parse a simple IDS string into an :class:`IDSTree`.

    The parser is intentionally minimal and assumes that all operators are
    binary.  It consumes characters from ``s`` sequentially and builds the
    corresponding tree.
    """

    return _parse_it(iter(s))


def is_valid_ids(t: IDSTree) -> bool:
    """Return ``True`` if ``t`` is structurally valid."""

    if t.value in IDS_TOKENS:
        if len(t.children) != 2:
            return False
        return all(is_valid_ids(c) for c in t.children)
    return t.value != ""


def contains_base(t: IDSTree, base: str) -> bool:
    """Check whether ``base`` appears as a leaf in ``t``."""

    if t.value == base:
        return True
    return any(contains_base(c, base) for c in t.children)


def layout_hint(t: IDSTree) -> str:
    """Return the top level layout operator for ``t``."""

    if t.value in IDS_TOKENS:
        return t.value
    for ch in t.children:
        hint = layout_hint(ch)
        if hint in IDS_TOKENS:
            return hint
    return "⿰"


def score_ids_legality(t: IDSTree) -> float:
    """Score ``t`` in ``[0, 1]`` based on structural legality."""

    return 1.0 if is_valid_ids(t) else 0.0


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point printing simple IDS diagnostics."""

    parser = argparse.ArgumentParser(description="IDS grammar helper")
    parser.add_argument("ids", help="IDS expression, e.g. '⿰木木'")
    parser.add_argument("--base", default="", help="Base character to test")
    args = parser.parse_args(list(argv) if argv is not None else None)

    tree = parse_ids(args.ids)
    print(tree)
    if args.base:
        print("contains_base", contains_base(tree, args.base))
    print("layout", layout_hint(tree))
    print("legality", score_ids_legality(tree))


if __name__ == "__main__":  # pragma: no cover
    main()

