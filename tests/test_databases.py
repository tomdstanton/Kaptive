r"""3-stage generic database validation: Stage-1 degradation, Stage-2 LOLO, and Stage-3 `test.tsv`."""

from __future__ import annotations

import pytest

from kaptive.db.manager import DatabaseManager
from kaptive.serotyping.core import Serotyper
from tests.curated import CuratedTestLoader, NcbiGenomeCache, SerotypeExpectation
from tests.toolkit import (
    LocusCrossValidator,
    LocusInjector,
    MaskedSerotyper,
    Stage1DegradationValidator,
)

KNOWN_KEYWORDS = tuple(DatabaseManager._KNOWN.keys())


@pytest.mark.integration
class TestDatabases:
    r"""Validate registered Kaptive databases across all 3 generic test stages."""

    @pytest.mark.parametrize(
        "db_keyword",
        [pytest.param(k, id=k, marks=pytest.mark.xdist_group(k)) for k in KNOWN_KEYWORDS],
    )
    def test_stage1_injection_gc_and_degradation(
        self,
        db_keyword: str,
        pytestconfig: pytest.Config,
    ) -> None:
        r"""Stage 1: 5 Mbp chromosome GC sweep + orthogonal fragmentation, dropout, and divergence ladders."""
        db = DatabaseManager.get(db_keyword)
        serotyper = Serotyper(db)

        stage1_all = bool(pytestconfig.getoption("--stage1-all-loci"))
        n_sample = None if stage1_all else int(pytestconfig.getoption("--loci-per-db"))
        sampled_indices = LocusInjector.sample_loci(db, n=n_sample)
        assert len(sampled_indices) > 0

        validator = Stage1DegradationValidator(seed=1000)
        validator.validate(db, serotyper, locus_indices=sampled_indices)

    @pytest.mark.parametrize(
        "db_keyword",
        [pytest.param(k, id=k, marks=pytest.mark.xdist_group(k)) for k in KNOWN_KEYWORDS],
    )
    def test_stage2_lolo_cross_validation(
        self,
        db_keyword: str,
        pytestconfig: pytest.Config,
    ) -> None:
        r"""Stage 2: Single-pass self-consistency and Leave-One-Locus-Out (LOLO) cross-validation."""
        db = DatabaseManager.get(db_keyword)
        serotyper = MaskedSerotyper(db)

        all_loci = bool(pytestconfig.getoption("--all-loci"))
        n_sample = None if all_loci else int(pytestconfig.getoption("--loci-per-db"))
        sampled_indices = LocusInjector.sample_loci(db, n=n_sample)
        assert len(sampled_indices) > 0

        allowlist = CuratedTestLoader.load_lolo_allowlist(db_keyword, include_remote=True)
        validator = LocusCrossValidator(allowlist=allowlist, seed=2000)
        validator.validate(db, serotyper, locus_indices=sampled_indices)

    def test_stage3_curated_tsv_expectation(
        self,
        curated_case: SerotypeExpectation,
        genome_cache: NcbiGenomeCache,
    ) -> None:
        r"""Stage 3: Decentralized curator `test.tsv` regression expectations."""
        db = DatabaseManager.get(curated_case.database)
        serotyper = Serotyper(db)
        assembly = genome_cache.resolve(curated_case)
        result = serotyper(assembly)
        curated_case.verify(result)
