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

import logging
from dataclasses import dataclass
from typing import Any, cast

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from sql_metadata.dialect_parser import DialectParser

__all__ = ["Edge", "LineageGraph", "StatementLineage", "build"]


# -- statement splitting ----------------------------------------------------


def _parse_all(sql: str, dialect: Any) -> "list[exp.Expression | None]":
    """Parse *sql* into a list of statement ASTs with one dialect.

    Mirrors :meth:`DialectParser._parse_with_dialect` (same ``WARN``
    error level and logger suppression) but keeps every statement of the
    batch instead of only the first one.
    """
    logger = logging.getLogger("sqlglot")
    old_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        return cast(
            "list[exp.Expression | None]",
            sqlglot.parse(sql, dialect=dialect, error_level=sqlglot.ErrorLevel.WARN),
        )
    finally:
        logger.setLevel(old_level)


def _split_statements(sql: str) -> "list[tuple[int, exp.Expression]]":
    """Split *sql* into ``(index, ast)`` pairs, one per statement.

    Reuses :class:`DialectParser` dialect detection and parse-quality
    checks so a batch is parsed exactly the way the rest of the library
    would parse its statements.  Empty and comment-only statements
    (which sqlglot reports as ``None``) and stray semicolons are
    skipped; the returned indices still reflect the original position
    in the batch.
    """
    if not sql or not sql.strip():
        return []
    dialects = DialectParser._detect_dialects(sql)
    parser = DialectParser()
    for dialect in dialects:
        try:
            results = _parse_all(sql, dialect)
        except (ParseError, TokenError):
            continue
        statements = [
            (index, result)
            for index, result in enumerate(results)
            if result is not None and not isinstance(result, exp.Semicolon)
        ]
        if not statements:
            continue
        is_last = dialect == dialects[-1]
        if not is_last and any(
            parser._is_degraded(result, sql) for _, result in statements
        ):
            continue
        return statements
    return []


# -- read/write determination ------------------------------------------------


def _target_table(node: "exp.Expression | None") -> "exp.Table | None":
    """Unwrap a DML target down to its :class:`~sqlglot.exp.Table` node."""
    while isinstance(node, exp.Schema):
        node = node.this
    return node if isinstance(node, exp.Table) else None


def _write_targets(ast: exp.Expression) -> "list[exp.Table]":
    """Physical tables written by the statement rooted at *ast*.

    * ``INSERT INTO t`` / ``UPDATE t`` / ``DELETE FROM t`` → ``t``
    * ``CREATE TABLE t ...`` (with or without ``AS SELECT``) → ``t``

    Anything else (plain ``SELECT``, ``DROP``, ``exp.Command``, …)
    writes nothing.
    """
    target: "exp.Expression | None" = None
    if isinstance(ast, (exp.Insert, exp.Update, exp.Delete)):
        target = ast.this
    elif isinstance(ast, exp.Create) and ast.kind == "TABLE":
        target = ast.this
    table = _target_table(target)
    return [table] if table is not None else []


def _cte_names(ast: exp.Expression) -> set[str]:
    """Lower-cased names of all CTEs defined in the statement."""
    return {cte.alias.lower() for cte in ast.find_all(exp.CTE)}


def _extract_reads_writes(
    ast: exp.Expression,
) -> "tuple[list[exp.Table], list[exp.Table]]":
    """Return ``(read_tables, write_tables)`` for one statement AST.

    Reads are every physical table referenced anywhere in the statement
    — ``FROM``/``JOIN`` clauses, ``USING``, subqueries in ``SET`` or
    ``WHERE``, CTE bodies — drilled all the way down.  References to
    CTE names are not physical tables and are excluded; the write
    targets themselves are not reads.
    """
    writes = _write_targets(ast)
    write_ids = {id(node) for node in writes}
    cte_names = _cte_names(ast)
    reads = [
        table
        for table in ast.find_all(exp.Table)
        if id(table) not in write_ids
        and not (table.name.lower() in cte_names and not table.db and not table.catalog)
    ]
    return reads, writes


# -- name normalisation ------------------------------------------------------


def _clean_part(part: str) -> str:
    """Normalise one identifier segment: strip quoting, lower-case."""
    return part.strip('`"[]').lower()


def _normalize_table(table: exp.Table) -> "tuple[str, ...] | None":
    """Normalised identity of a physical table, or ``None`` to skip it.

    The identity is a tuple of segments — ``("schema", "table")`` for a
    qualified name, ``("table",)`` otherwise — so segments are kept
    separate and never joined into a single string.  Quoting styles
    (back-ticks, double quotes, square brackets) and case are spelling
    details and are erased.  Temporary tables (``#temp``) never enter
    the graph.
    """
    parts = [p for p in (table.catalog, table.db, table.name) if p]
    key = tuple(c for c in (_clean_part(p) for p in parts) if c)
    if not key:
        return None
    if key[-1].startswith("#"):
        return None
    return key


def _display_name(table: exp.Table) -> str:
    """Original spelling of a table name, kept for display only."""
    return ".".join(p for p in (table.catalog, table.db, table.name) if p)


def _normalize_name(table: "tuple[str, ...] | str") -> tuple[str, ...]:
    """Normalise a caller-supplied table name to its node identity."""
    parts = table.split(".") if isinstance(table, str) else list(table)
    return tuple(_clean_part(p) for p in parts if _clean_part(p))


# -- graph algorithms ---------------------------------------------------------


def _adjacency(
    nodes: "list[tuple[str, ...]]", edges: "list[Edge]"
) -> "dict[tuple[str, ...], set[tuple[str, ...]]]":
    """Directed adjacency of the graph: written table → read table."""
    adjacency: dict[tuple[str, ...], set[tuple[str, ...]]] = {
        node: set() for node in nodes
    }
    for edge in edges:
        adjacency[edge.written].add(edge.read)
    return adjacency


def _strongly_connected_components(  # noqa: C901 — iterative Tarjan is inherently branchy
    nodes: "list[tuple[str, ...]]",
    adjacency: "dict[tuple[str, ...], set[tuple[str, ...]]]",
) -> "list[list[tuple[str, ...]]]":
    """Tarjan's algorithm (iterative) over the ``written -> read`` edges."""
    index_of: dict[tuple[str, ...], int] = {}
    lowlink: dict[tuple[str, ...], int] = {}
    stack: list[tuple[str, ...]] = []
    on_stack: set[tuple[str, ...]] = set()
    components: list[list[tuple[str, ...]]] = []
    counter = 0
    for root in nodes:
        if root in index_of:
            continue
        index_of[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        call_stack = [(root, iter(adjacency[root]))]
        while call_stack:
            node, successors = call_stack[-1]
            descended = False
            for succ in successors:
                if succ not in index_of:
                    index_of[succ] = lowlink[succ] = counter
                    counter += 1
                    stack.append(succ)
                    on_stack.add(succ)
                    call_stack.append((succ, iter(adjacency[succ])))
                    descended = True
                    break
                if succ in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[succ])
            if descended:
                continue
            call_stack.pop()
            if call_stack:
                parent = call_stack[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index_of[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    return components


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

    def _register(self, key: tuple[str, ...], display: str) -> None:
        """Record *display* as the first-seen spelling of *key*."""
        if key not in self._originals:
            self._originals[key] = display

    def _keys_for(self, tables: "list[exp.Table]") -> "list[tuple[str, ...]]":
        """Normalise table nodes to unique keys, registering spellings."""
        keys: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for table in tables:
            key = _normalize_table(table)
            if key is None or key in seen:
                continue
            seen.add(key)
            keys.append(key)
            self._register(key, _display_name(table))
        return keys

    @property
    def nodes(self) -> list[tuple[str, ...]]:
        """All normalised table names in the graph, in first-seen order."""
        return list(self._originals)

    @property
    def edges(self) -> list[Edge]:
        """All read/write edges, in statement order.

        Every statement that both reads and writes contributes one edge
        per ``(written, read)`` pair, pointing from the written table to
        the table it was read from.  When two statements between the
        same pair of tables go opposite ways, both edges are kept.
        """
        result: list[Edge] = []
        for statement in self._statements:
            for written in statement.writes:
                for read in statement.reads:
                    result.append(
                        Edge(written=written, read=read, statement=statement.index)
                    )
        return result

    @property
    def statements(self) -> list[StatementLineage]:
        """Per-statement read/write sets."""
        return self._statements

    def original_name(self, node: "tuple[str, ...] | str") -> str:
        """First-seen original spelling of *node* (display only)."""
        return self._originals[_normalize_name(node)]

    def upstream(self, table: "tuple[str, ...] | str") -> list[tuple[str, ...]]:
        """Tables read by statements that wrote *table* (one hop)."""
        key = _normalize_name(table)
        seen: set[tuple[str, ...]] = set()
        result: list[tuple[str, ...]] = []
        for edge in self.edges:
            if edge.written == key and edge.read not in seen:
                seen.add(edge.read)
                result.append(edge.read)
        return result

    def downstream(self, table: "tuple[str, ...] | str") -> list[tuple[str, ...]]:
        """Tables written by statements that read *table* (one hop)."""
        key = _normalize_name(table)
        seen: set[tuple[str, ...]] = set()
        result: list[tuple[str, ...]] = []
        for edge in self.edges:
            if edge.read == key and edge.written not in seen:
                seen.add(edge.written)
                result.append(edge.written)
        return result

    @property
    def topological_order(self) -> list[tuple[str, ...]]:
        """Nodes ordered written-before-read; cycles do not break the order.

        Follows the edge direction (written table before the tables it
        was read from).  When the remaining nodes form a cycle, the
        earliest-appearing node is emitted first and the walk continues,
        so a cyclic batch still yields every node exactly once.
        """
        nodes = self.nodes
        rank = {node: position for position, node in enumerate(nodes)}
        incoming: dict[tuple[str, ...], set[tuple[str, ...]]] = {
            node: set() for node in nodes
        }
        for edge in self.edges:
            incoming[edge.read].add(edge.written)
        remaining = set(nodes)
        order: list[tuple[str, ...]] = []
        while remaining:
            ready = [n for n in remaining if not incoming[n] & remaining]
            # no ready node -> cycle: break it at the earliest-appearing node
            pick = (
                min(ready, key=rank.__getitem__)
                if ready
                else min(remaining, key=rank.__getitem__)
            )
            order.append(pick)
            remaining.discard(pick)
        return order

    @property
    def cycles(self) -> list[list[tuple[str, ...]]]:
        """Cycles in the graph; members listed in first-seen order.

        A cycle is a strongly connected component of more than one
        table, or a single table that a statement both reads and writes.
        """
        nodes = self.nodes
        rank = {node: position for position, node in enumerate(nodes)}
        adjacency = _adjacency(nodes, self.edges)
        result: list[list[tuple[str, ...]]] = []
        for component in _strongly_connected_components(nodes, adjacency):
            if len(component) > 1:
                result.append(sorted(component, key=rank.__getitem__))
            elif component[0] in adjacency[component[0]]:
                result.append(component)
        result.sort(key=lambda cycle: min(rank[node] for node in cycle))
        return result


def build(sql: str) -> LineageGraph:
    """Build a :class:`LineageGraph` from a batch of SQL statements.

    :param sql: One or more SQL statements separated by semicolons.
    :returns: The lineage graph for the batch.
    """
    graph = LineageGraph()
    for index, ast in _split_statements(sql):
        reads, writes = _extract_reads_writes(ast)
        # writes are registered before reads so first-seen order matches
        # the textual order of the statement (target precedes sources)
        write_keys = graph._keys_for(writes)
        read_keys = graph._keys_for(reads)
        if not read_keys and not write_keys:
            continue
        graph._statements.append(
            StatementLineage(
                index=index, reads=tuple(read_keys), writes=tuple(write_keys)
            )
        )
    return graph
