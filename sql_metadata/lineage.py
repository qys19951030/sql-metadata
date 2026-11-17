"""Cross-statement table lineage for batches of SQL scripts.

Where :class:`~sql_metadata.parser.Parser` analyses a single statement,
this module analyses a *batch* of statements and extracts the read/write
relationships between physical tables:

* ``SELECT`` reads the tables in its ``FROM``/``JOIN`` clauses.
* ``INSERT INTO t ...`` writes ``t``; a following ``SELECT`` contributes reads.
* ``UPDATE t ...`` writes ``t``; ``FROM`` and subqueries in ``SET`` are reads.
* ``DELETE FROM t ...`` writes ``t``; ``USING`` tables are reads.
* ``CREATE TABLE t AS SELECT ...`` writes ``t``; the query body is read.

CTE names are not physical tables and never become nodes, but physical
tables referenced *inside* a CTE body count as reads of the statement.
Subqueries are followed all the way down.  Temporary tables, ``VALUES``
clauses and function calls never enter the graph.

Table name normalisation: quoting (back-ticks, double quotes, square
brackets) is a spelling detail and is stripped; names are lower-cased;
a qualified ``schema.table`` keeps both segments as a tuple
``("schema", "table")`` — segments are never joined into a single
string.  The first-seen original spelling is kept on the node for
display only and never participates in comparison.

Edge direction convention: an edge points from the *written* table to
the *read* table (``written -> read``), so :attr:`LineageGraph.topological_order`
lists written tables before the tables they were read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlglot import exp

from sql_metadata.dialect_parser import DialectParser

__all__ = ["Edge", "LineageGraph", "StatementLineage", "build"]


@dataclass(frozen=True)
class Edge:
    """A single read/write relationship between two tables.

    ``written`` is the table a statement wrote to, ``read`` is a table
    the same statement read from, and ``statement`` is the 0-based index
    of that statement in the batch, so every edge can be traced back to
    the script it came from.
    """

    written: tuple[str, ...]
    read: tuple[str, ...]
    statement: int


@dataclass(frozen=True)
class StatementLineage:
    """Read/write sets of one statement in the batch."""

    index: int
    reads: tuple[tuple[str, ...], ...]
    writes: tuple[tuple[str, ...], ...]


class LineageGraph:
    """Table-level lineage graph built from a batch of SQL statements."""

    def __init__(self) -> None:
        self._statements: list[StatementLineage] = []
        self._originals: dict[tuple[str, ...], str] = {}

    @property
    def nodes(self) -> list[tuple[str, ...]]:
        """All normalised table names in the graph, in first-seen order."""
        raise NotImplementedError

    @property
    def edges(self) -> list[Edge]:
        """All read/write edges, in statement order."""
        raise NotImplementedError

    @property
    def statements(self) -> list[StatementLineage]:
        """Per-statement read/write sets."""
        return self._statements

    def original_name(self, node: tuple[str, ...]) -> str:
        """First-seen original spelling of *node* (display only)."""
        raise NotImplementedError

    def upstream(self, table: "tuple[str, ...] | str") -> list[tuple[str, ...]]:
        """Tables read by statements that wrote *table* (one hop)."""
        raise NotImplementedError

    def downstream(self, table: "tuple[str, ...] | str") -> list[tuple[str, ...]]:
        """Tables written by statements that read *table* (one hop)."""
        raise NotImplementedError

    @property
    def topological_order(self) -> list[tuple[str, ...]]:
        """Nodes ordered written-before-read; cycles do not break the order."""
        raise NotImplementedError

    @property
    def cycles(self) -> list[list[tuple[str, ...]]]:
        """Cycles in the graph; members listed in first-seen order."""
        raise NotImplementedError


def build(sql: str) -> LineageGraph:
    """Build a :class:`LineageGraph` from a batch of SQL statements.

    :param sql: One or more SQL statements separated by semicolons.
    :returns: The lineage graph for the batch.
    """
    raise NotImplementedError
