"""Tests for sql_metadata.lineage — cross-statement table lineage."""

from sql_metadata.lineage import (
    Edge,
    LineageGraph,
    StatementLineage,
    _extract_reads_writes,
    _split_statements,
    build,
)


def test_module_exports():
    # Skeleton stage: the public names exist and are importable.
    assert callable(build)
    assert LineageGraph is not None
    assert Edge is not None
    assert StatementLineage is not None


# -- statement splitting -----------------------------------------------------


def _names(tables):
    return sorted(t.name for t in tables)


def test_split_multiple_statements():
    statements = _split_statements("SELECT 1; SELECT 2; SELECT 3")
    assert [index for index, _ in statements] == [0, 1, 2]


def test_split_skips_empty_and_comment_only_statements():
    statements = _split_statements(
        "SELECT 1;;\n-- nothing here\nSELECT 2\n-- trailing comment"
    )
    assert len(statements) == 2
    # indices still reflect the original position in the batch
    assert [index for index, _ in statements] == [0, 2]


def test_split_empty_input():
    assert _split_statements("") == []
    assert _split_statements("   \n  ") == []
    assert _split_statements("-- only a comment") == []


# -- read/write determination ------------------------------------------------


def _reads_writes(sql):
    _, ast = _split_statements(sql)[0]
    return _extract_reads_writes(ast)


def test_select_reads_from_and_join():
    reads, writes = _reads_writes("SELECT * FROM a JOIN b ON a.id = b.id")
    assert _names(reads) == ["a", "b"]
    assert writes == []


def test_insert_select_writes_target_reads_sources():
    reads, writes = _reads_writes("INSERT INTO t SELECT * FROM s")
    assert _names(writes) == ["t"]
    assert _names(reads) == ["s"]


def test_insert_values_writes_target_only():
    reads, writes = _reads_writes("INSERT INTO t (a) VALUES (1)")
    assert _names(writes) == ["t"]
    assert reads == []


def test_update_reads_from_and_set_subquery():
    reads, writes = _reads_writes(
        "UPDATE t SET a = (SELECT max(x) FROM s) FROM u WHERE t.id = u.id"
    )
    assert _names(writes) == ["t"]
    assert _names(reads) == ["s", "u"]


def test_delete_reads_using():
    reads, writes = _reads_writes("DELETE FROM t USING u WHERE t.id = u.id")
    assert _names(writes) == ["t"]
    assert _names(reads) == ["u"]


def test_create_table_as_select():
    reads, writes = _reads_writes("CREATE TABLE t AS SELECT * FROM s")
    assert _names(writes) == ["t"]
    assert _names(reads) == ["s"]


def test_cte_name_is_not_a_table_but_its_body_is_read():
    reads, writes = _reads_writes(
        "WITH cte AS (SELECT * FROM physical) SELECT * FROM cte JOIN other ON 1 = 1"
    )
    assert writes == []
    assert _names(reads) == ["other", "physical"]


def test_subqueries_are_followed_all_the_way_down():
    reads, writes = _reads_writes(
        "SELECT * FROM a WHERE id IN (SELECT id FROM (SELECT id FROM deep) d)"
    )
    assert _names(reads) == ["a", "deep"]


def test_same_table_read_and_written_kept_on_both_sides():
    reads, writes = _reads_writes("UPDATE t SET x = (SELECT max(x) FROM t)")
    assert _names(writes) == ["t"]
    assert _names(reads) == ["t"]


# -- name normalisation and node merging --------------------------------------


def test_same_table_different_spellings_merge_into_one_node():
    graph = build(
        "SELECT * FROM Users;"
        'SELECT * FROM "USERS";'
        "SELECT * FROM `users`"
    )
    assert graph.nodes == [("users",)]


def test_schema_prefix_kept_as_separate_segments():
    graph = build("SELECT * FROM Public.Users")
    assert graph.nodes == [("public", "users")]
    # a qualified name and a bare name are different nodes
    graph = build("SELECT * FROM Public.Users; SELECT * FROM users")
    assert graph.nodes == [("public", "users"), ("users",)]


def test_quoting_styles_normalised():
    # each quoting style is parsed by its own dialect, but all of them
    # must normalise to the same node identity
    assert build("SELECT * FROM `Sch`.`Tab`").nodes == [("sch", "tab")]
    assert build('SELECT * FROM "Sch"."Tab"').nodes == [("sch", "tab")]
    assert build("SELECT * FROM [Sch].[Tab]").nodes == [("sch", "tab")]


def test_original_spelling_kept_for_display_only():
    graph = build('SELECT * FROM `Public`.`Users`; SELECT * FROM "public"."users"')
    assert graph.original_name(("public", "users")) == "Public.Users"
    # lookup accepts a plain string too
    assert graph.original_name("public.users") == "Public.Users"


def test_temp_tables_never_enter_the_graph():
    graph = build(
        "SELECT * INTO #tmp FROM real_table;"
        "SELECT * FROM #tmp t JOIN other_table o ON t.id = o.id"
    )
    assert ("#tmp",) not in graph.nodes
    assert ("tmp",) not in graph.nodes


def test_empty_and_comment_only_statements_produce_no_nodes():
    graph = build(";\n-- just a comment\n;\nSELECT * FROM t")
    assert graph.nodes == [("t",)]
    assert [s.index for s in graph.statements] == [3]


def test_statement_lineage_records_reads_and_writes():
    graph = build("INSERT INTO t SELECT * FROM s; SELECT * FROM t")
    first, second = graph.statements
    assert first.index == 0
    assert first.writes == (("t",),)
    assert first.reads == (("s",),)
    assert second.reads == (("t",),)
    assert second.writes == ()
