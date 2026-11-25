r"""Pytest configuration, xdist/Numba thread pinning, CLI flags, and dynamic test.tsv parametrization."""

from __future__ import annotations

import os
from pathlib import Path

if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import pytest

from tests.curated import CuratedTestLoader, NcbiGenomeCache


def pytest_addoption(parser: pytest.Parser) -> None:
    r"""Register Kaptive database & integration testing CLI options."""
    group = parser.getgroup("kaptive", "Kaptive decentralized database and simulation options")
    group.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests across installed/remote databases and curated test.tsv genomes.",
    )
    group.addoption(
        "--all-loci",
        action="store_true",
        default=False,
        help="Test 100%% of loci in every database instead of a stratified sample.",
    )
    group.addoption(
        "--loci-per-db",
        type=int,
        default=5,
        help="Number of loci to sample per database when --all-loci is not set (default: 5).",
    )
    group.addoption(
        "--db-repo",
        action="append",
        default=[],
        metavar="DIR",
        help="Local clone path of a database repository containing .gbk/.toml and optional tests/test.tsv.",
    )
    group.addoption(
        "--test-tsv",
        action="append",
        default=[],
        metavar="FILE",
        help="Explicit path to a curated test.tsv expectation table.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    r"""Skip `@pytest.mark.integration` tests unless `--integration` is passed."""
    if config.getoption("--integration"):
        return
    skip_marker = pytest.mark.skip(reason="Pass --integration to run database & curated assembly tests.")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    r"""Dynamically parametrize `curated_case` from `--test-tsv`, `--db-repo`, and remote repos."""
    if "curated_case" not in metafunc.fixturenames:
        return

    run_integration = bool(metafunc.config.getoption("--integration"))
    tsv_paths = tuple(Path(p).expanduser().resolve() for p in metafunc.config.getoption("--test-tsv"))
    repo_paths = tuple(Path(p).expanduser().resolve() for p in metafunc.config.getoption("--db-repo"))
    cases = CuratedTestLoader.discover_all(
        tsv_files=tsv_paths,
        db_repos=repo_paths,
        include_remote=run_integration,
    )
    params = [
        pytest.param(case, id=case.test_id, marks=pytest.mark.xdist_group(case.database))
        for case in cases
    ]
    metafunc.parametrize("curated_case", params)


@pytest.fixture(scope="session")
def genome_cache() -> NcbiGenomeCache:
    r"""Session-scoped `NcbiGenomeCache` instance for resolving NCBI accessions and local FASTAs."""
    return NcbiGenomeCache()
