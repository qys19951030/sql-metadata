"""Tests for sql_metadata.lineage — cross-statement table lineage."""

from sql_metadata.lineage import Edge, LineageGraph, StatementLineage, build


def test_module_exports():
    # Skeleton stage: the public names exist and are importable.
    assert callable(build)
    assert LineageGraph is not None
    assert Edge is not None
    assert StatementLineage is not None
