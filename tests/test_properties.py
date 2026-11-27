r"""Property-based Hypothesis tests for Kaptive `core/` SoA containers and `tests/toolkit.py`."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from kaptive.core.alignment import CigarOp, Cigars
from kaptive.core.genome import GenomeAssembly
from kaptive.core.interval import Intervals
from kaptive.core.pairwise import PairwiseAligner
from kaptive.core.seq import Sequences
from kaptive.serotyping.models import SerotypingProblem
from tests.curated import CuratedTestLoader, NcbiGenomeCache, SerotypeExpectation
from tests.toolkit import LocusCrossValidator, SequenceGenerator, SequenceShredder


class TestCoreSoAProperties:
    r"""Algebraic invariants across ragged `Sequences`, `Intervals`, `Cigars`, and `PairwiseAligner`."""

    @given(
        gc=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        lengths=st.lists(st.integers(min_value=0, max_value=2_000), min_size=1, max_size=12),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_sequences_slicing_rc_involution_and_fasta_roundtrip(
        self,
        gc: float,
        lengths: list[int],
        seed: int,
    ) -> None:
        batch = SequenceGenerator(gc=gc, seed=seed).generate(lengths, prefix="s")
        assert len(batch) == len(lengths)
        np.testing.assert_array_equal(batch.lengths, np.array(lengths, dtype=np.int32))

        # Double reverse-complement extraction is an exact involution
        n = len(batch)
        indices = np.arange(n, dtype=np.int32)
        starts = np.zeros(n, dtype=np.int32)
        ends = batch.lengths.astype(np.int32)
        neg_strands = np.full(n, -1, dtype=np.int8)

        rc_once = batch.extract(indices, starts, ends, neg_strands, new_ids=batch.ids)
        rc_twice = rc_once.extract(indices, starts, ends, neg_strands, new_ids=batch.ids)
        np.testing.assert_array_equal(rc_twice.seqs, batch.seqs)
        np.testing.assert_array_equal(rc_twice.lengths, batch.lengths)

        # Boolean mask vs integer index slicing equivalence
        mask = (np.arange(n) % 2) == 0
        int_idx = np.flatnonzero(mask)
        by_mask = batch[mask]
        by_int = batch[int_idx]
        assert by_mask.ids == by_int.ids
        np.testing.assert_array_equal(by_mask.seqs, by_int.seqs)

        # FASTA stream round-trip for non-empty sequences
        non_empty = batch[batch.lengths > 0]
        if len(non_empty) > 0:
            asm = GenomeAssembly.from_stream(io.BytesIO(non_empty.to_fasta()))
            assert asm.contigs.ids == non_empty.ids
            np.testing.assert_array_equal(asm.contigs.seqs, non_empty.seqs)

    @given(
        lengths=st.lists(st.integers(min_value=3, max_value=900), min_size=1, max_size=8),
        frame=st.integers(min_value=0, max_value=2),
        to_stop=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_sequences_translate_bounds(
        self,
        lengths: list[int],
        frame: int,
        to_stop: bool,
        seed: int,
    ) -> None:
        batch = SequenceGenerator(gc=0.5, seed=seed).generate(lengths)
        frames = np.full(len(batch), frame, dtype=np.int32)
        proteins = batch.translate(frames=frames, to_stop=to_stop)
        assert len(proteins) == len(batch)
        max_codons = np.maximum(0, (batch.lengths - frame) // 3)
        if to_stop:
            assert bool(np.all((proteins.lengths >= 0) & (proteins.lengths <= max_codons)))
        else:
            np.testing.assert_array_equal(proteins.lengths, max_codons)

    @given(
        raw_coords=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=5_000),
                st.integers(min_value=1, max_value=1_000),
                st.integers(min_value=0, max_value=2),
            ),
            min_size=1,
            max_size=25,
        ),
        max_ov=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        tolerance=st.integers(min_value=0, max_value=500),
    )
    def test_intervals_cull_overlaps_and_spatial_clustering(
        self,
        raw_coords: list[tuple[int, int, int]],
        max_ov: float,
        tolerance: int,
    ) -> None:
        starts = np.array([s for s, _, _ in raw_coords], dtype=np.int32)
        ends = np.array([s + span for s, span, _ in raw_coords], dtype=np.int32)
        groups = np.array([g for _, _, g in raw_coords], dtype=np.int32)
        strands = np.ones(len(raw_coords), dtype=np.int8)

        iv = Intervals(starts=starts, ends=ends, strands=strands)
        order = np.arange(len(iv), dtype=np.int32)
        kept_mask = iv.cull_overlaps(order=order, max_overlap_fraction=max_ov, group_by=groups)
        assert kept_mask.shape == (len(iv),)
        assert bool(np.any(kept_mask))

        # Pairwise invariant: no two kept intervals in the same group exceed max_ov
        kept_idx = np.flatnonzero(kept_mask)
        for i_pos in range(len(kept_idx)):
            i = int(kept_idx[i_pos])
            for j_pos in range(i_pos + 1, len(kept_idx)):
                j = int(kept_idx[j_pos])
                if groups[i] != groups[j]:
                    continue
                ov = min(int(ends[i]), int(ends[j])) - max(int(starts[i]), int(starts[j]))
                if ov > 0:
                    shorter = min(int(ends[i] - starts[i]), int(ends[j] - starts[j]))
                    assert (ov / shorter) <= max_ov + 1e-9

        # Spatial clustering: distinct groups never share a cluster ID
        clusters = iv.cluster_spatial(tolerance=tolerance, group_by=groups)
        for i in range(len(iv)):
            for j in range(i + 1, len(iv)):
                if groups[i] != groups[j]:
                    assert int(clusters[i]) != int(clusters[j])

    @given(
        cigar_specs=st.lists(
            st.lists(
                st.tuples(
                    st.integers(min_value=1, max_value=200),
                    st.sampled_from([int(CigarOp.M), int(CigarOp.I), int(CigarOp.D), int(CigarOp.S)]),
                ),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=10,
        )
    )
    def test_cigars_swap_sides_involution(self, cigar_specs: list[list[tuple[int, int]]]) -> None:
        arrays = [
            np.array([(length << 4) | op for length, op in ops], dtype=np.uint32)
            for ops in cigar_specs
        ]
        cigars = Cigars.from_lists(arrays)
        swapped_twice = cigars.swap_sides().swap_sides()
        np.testing.assert_array_equal(swapped_twice.data, cigars.data)
        np.testing.assert_array_equal(swapped_twice.offsets, cigars.offsets)
        np.testing.assert_array_equal(swapped_twice.lengths, cigars.lengths)

    @given(
        lengths=st.lists(st.integers(min_value=30, max_value=300), min_size=1, max_size=6),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_pairwise_aligner_self_identity_and_bounds(
        self,
        lengths: list[int],
        seed: int,
    ) -> None:
        nt_a = SequenceGenerator(gc=0.5, seed=seed).generate(lengths)
        nt_b = SequenceGenerator(gc=0.5, seed=seed + 1).generate(lengths)
        prot_a = nt_a.translate(to_stop=False)
        prot_b = nt_b.translate(to_stop=False)

        aligner = PairwiseAligner()
        self_alns = aligner(prot_a, prot_a)
        np.testing.assert_allclose(self_alns.pidents, 100.0)

        cross_alns = aligner(prot_a, prot_b)
        assert bool(np.all((cross_alns.pidents >= 0.0) & (cross_alns.pidents <= 100.0)))


class TestToolkitProperties:
    r"""Property-based invariants for `SequenceGenerator`, `SequenceShredder`, and `CuratedTestLoader`."""

    @given(
        gc=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        length=st.integers(min_value=4_000, max_value=15_000),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_sequence_generator_gc_concentration_and_read_only(
        self,
        gc: float,
        length: int,
        seed: int,
    ) -> None:
        gen = SequenceGenerator(gc=gc, seed=seed)
        b1 = gen.generate(length)
        b2 = gen.generate(length)

        assert not b1.seqs.flags.writeable
        assert not b1.offsets.flags.writeable
        assert not b1.lengths.flags.writeable
        np.testing.assert_array_equal(b1.seqs, b2.seqs)

        valid_ascii = np.isin(b1.seqs, np.array([65, 67, 71, 84], dtype=np.uint8))
        assert bool(np.all(valid_ascii))

        observed_gc = float(np.mean((b1.seqs == 67) | (b1.seqs == 71)))
        assert observed_gc == pytest.approx(gc, abs=0.04)

    @given(
        lengths=st.lists(st.integers(min_value=0, max_value=6_000), min_size=0, max_size=8),
        n_fragments=st.integers(min_value=1, max_value=10),
        gap_len=st.integers(min_value=0, max_value=250),
        flip_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        snp_rate=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        indel_rate=st.floats(min_value=0.0, max_value=0.2, allow_nan=False),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_sequence_shredder_arbitrary_parameters_never_corrupt_soa(
        self,
        lengths: list[int],
        n_fragments: int,
        gap_len: int,
        flip_prob: float,
        snp_rate: float,
        indel_rate: float,
        seed: int,
    ) -> None:
        parent = SequenceGenerator(gc=0.5, seed=seed).generate(lengths)
        shredder = SequenceShredder(
            n_fragments=n_fragments,
            gap_len=gap_len,
            flip_strand_prob=flip_prob,
            snp_rate=snp_rate,
            indel_rate=indel_rate,
            seed=seed + 7,
        )
        out = shredder.shred(parent)
        if len(out) > 0:
            assert not out.seqs.flags.writeable
            assert int(np.sum(out.lengths)) == len(out.seqs)
            assert bool(np.all(out.lengths > 0))
            valid_ascii = np.isin(out.seqs, np.array([65, 67, 71, 84], dtype=np.uint8))
            assert bool(np.all(valid_ascii))
            if gap_len == 0 and indel_rate == 0.0:
                assert int(np.sum(out.lengths)) == int(np.sum(parent.lengths))

    @given(flag_bits=st.integers(min_value=0, max_value=31))
    def test_problem_symbol_bijection(self, flag_bits: int) -> None:
        flags = SerotypingProblem(flag_bits)
        symbols = flags.to_symbols().decode("ascii")
        parsed = SerotypeExpectation.parse_problems(symbols or "none")
        assert parsed == flags

    def test_curated_tsv_and_allowlist_parsing(self, tmp_path: Path) -> None:
        tsv = (
            "# Regression table\n"
            "genome\tdatabase\tlocus\tphenotype\ttypeable\tproblems\tmissing_genes\tmin_coverage\n"
            "GCF_000000001.1\tdb_0\tlocus_0\ttype_0\ttrue\tnone\tnone\t98.5\n"
            "sample.fna.gz\tdb_0\tlocus_1\ttype_1\tfalse\t?!\tgene_04\t85.0\n"
        )
        cases = CuratedTestLoader.parse_tsv_text(tsv, source="inline")
        assert len(cases) == 2
        assert NcbiGenomeCache.is_ncbi_accession(cases[0].genome) is True
        assert NcbiGenomeCache.is_ncbi_accession(cases[1].genome) is False
        assert cases[0].missing_genes == ()
        assert cases[1].problems == SerotypeExpectation.parse_problems("FRAGMENTED|TRUNCATED_GENES")
        assert cases[1].missing_genes == ("gene_04",)

        toml_file = tmp_path / "db.toml"
        toml_file.write_text('[lolo_allowlist]\n"sub_*" = ["sub_*"]\n', encoding="utf-8")
        rules = CuratedTestLoader.load_lolo_allowlist("db_0", db_toml=toml_file)
        validator = LocusCrossValidator(allowlist=rules)
        assert validator.is_allowed_cross_hit("sub_1", "sub_2") is True
        assert validator.is_allowed_cross_hit("sub_1", "other_1") is False
