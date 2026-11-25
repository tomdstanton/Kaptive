r"""Class-first NumPy/Numba simulation, shredding, and LOLO cross-validation toolkit.

Follows the SeroVerse zero-copy Structure-of-Arrays (SoA) and strict class-first paradigms:
all random sequence generation, locus selection/injection, batch sequence shredding, and
single-pass Leave-One-Locus-Out (LOLO) evaluation live on immutable classes backed by
stateless SplitMix64 Numba kernels with zero module-level functions.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import rammappy

from kaptive.core._numba import njit, prange
from kaptive.core.alignment import Alignments
from kaptive.core.genome import GenomeAssembly
from kaptive.core.seq import BacterialTranslationTable, SeqRecord, Sequences
from kaptive.db.core import Database
from kaptive.serotyping.core import Serotyper
from kaptive.serotyping.models import SerotypingProblem, SerotypingResult

LocusFilter: TypeAlias = int | str | Sequence[int | str] | npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SequenceGenerator:
    r"""Numba-accelerated random nucleotide SoA batch generator.

    Uses per-sequence stateless SplitMix64 streams inside a parallel `@njit` kernel so output
    batches are deterministic regardless of thread count while writing directly into contiguous
    `Sequences` `uint8` buffers.

    Attributes:
        gc (float): Target GC fraction in `[0.0, 1.0]`. Defaults to `0.5`.
        seed (int): Deterministic 64-bit PRNG seed. Defaults to `42`.
    """

    gc: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        r"""Validate GC fraction bounds."""
        if not (0.0 <= self.gc <= 1.0):
            raise ValueError(f"gc must be between 0.0 and 1.0, got {self.gc}")

    @staticmethod
    @njit(parallel=True, cache=True, nogil=True)
    def _generate_dna_kernel(
        offsets: npt.NDArray[np.int32],
        lengths: npt.NDArray[np.int32],
        gc: float,
        seed: int,
    ) -> npt.NDArray[np.uint8]:
        n_seqs = len(lengths)
        total_len = int(offsets[n_seqs - 1]) + int(lengths[n_seqs - 1]) if n_seqs > 0 else 0
        out = np.empty(total_len, dtype=np.uint8)
        if total_len == 0:
            return out

        # Cumulative thresholds on [0, 1):
        # P(A) = (1 - gc)/2, P(C) = gc/2 => A+C = 0.5; P(G) = gc/2, P(T) = (1 - gc)/2
        thresh_a = (1.0 - gc) * 0.5
        thresh_c = 0.5
        thresh_g = 0.5 + gc * 0.5
        inv_53 = 1.0 / 9007199254740992.0
        golden = np.uint64(0x9E3779B97F4A7C15)
        mul1 = np.uint64(0xBF58476D1CE4E5B9)
        mul2 = np.uint64(0x94D049BB133111EB)
        base_seed = np.uint64(seed)

        for idx in prange(n_seqs):
            seq_len = int(lengths[idx])
            start = int(offsets[idx])
            state = base_seed ^ (np.uint64(idx + 1) * golden)
            for j in range(seq_len):
                state = state + golden
                z = state
                z = (z ^ (z >> np.uint64(30))) * mul1
                z = (z ^ (z >> np.uint64(27))) * mul2
                z = z ^ (z >> np.uint64(31))
                u = float(z >> np.uint64(11)) * inv_53
                if u < thresh_a:
                    out[start + j] = 65  # 'A'
                elif u < thresh_c:
                    out[start + j] = 67  # 'C'
                elif u < thresh_g:
                    out[start + j] = 71  # 'G'
                else:
                    out[start + j] = 84  # 'T'
        return out

    @staticmethod
    def _normalize_lengths(lengths: int | Sequence[int] | npt.NDArray[np.integer]) -> npt.NDArray[np.int32]:
        arr = (
            np.array([int(lengths)], dtype=np.int32)
            if isinstance(lengths, (int, np.integer))
            else np.asarray(lengths, dtype=np.int32).copy()
        )
        if arr.ndim != 1 or np.any(arr < 0):
            raise ValueError("lengths must be a non-negative int or 1D sequence of non-negative ints")
        arr.flags.writeable = False
        return arr

    @staticmethod
    def compute_offsets(lengths: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        r"""Compute read-only 0-based cumulative start offsets for a 1D `lengths` array."""
        offsets = np.zeros(len(lengths), dtype=np.int32)
        if len(lengths) > 1:
            np.cumsum(lengths[:-1], out=offsets[1:])
        offsets.flags.writeable = False
        return offsets

    def generate(
        self,
        lengths: int | Sequence[int] | npt.NDArray[np.integer],
        *,
        ids: Sequence[str] | None = None,
        prefix: str = "ctg",
    ) -> Sequences:
        r"""Generate a GC-weighted random nucleotide `Sequences` SoA batch."""
        len_arr = self._normalize_lengths(lengths)
        if len(len_arr) == 0:
            return Sequences.empty()
        id_tuple = tuple(ids) if ids is not None else tuple(f"{prefix}_{i}" for i in range(len(len_arr)))
        if len(id_tuple) != len(len_arr):
            raise ValueError(f"ids length ({len(id_tuple)}) must match lengths ({len(len_arr)})")
        offsets = self.compute_offsets(len_arr)
        seqs = self._generate_dna_kernel(offsets, len_arr, float(self.gc), int(self.seed))
        seqs.flags.writeable = False
        return Sequences(ids=id_tuple, seqs=seqs, offsets=offsets, lengths=len_arr)


@dataclass(frozen=True, slots=True)
class InjectedGenome:
    r"""Immutable record pairing a synthetic `GenomeAssembly` with the reference locus injected into it."""

    assembly: GenomeAssembly
    database_keyword: str
    locus_idx: int
    locus_name: str
    strand: int
    expected_phenotype: str


@dataclass(frozen=True, slots=True)
class LocusInjector:
    r"""Selects reference loci from any `Database` and injects them into GC-weighted synthetic assemblies."""

    gc: float = 0.5
    flank_length: int = 10_000
    background_contig_lengths: tuple[int, ...] = (15_000,)
    seed: int = 42

    @staticmethod
    def valid_locus_indices(db: Database) -> npt.NDArray[np.int32]:
        r"""Return read-only 0-based indices of all non-`Extra genes` loci in `db`."""
        expected_per_locus = np.zeros(len(db.loci), dtype=np.int32)
        np.add.at(expected_per_locus, db.gene_locus_indices[~db.extra_genes], 1)
        valid = np.flatnonzero(expected_per_locus > 0).astype(np.int32)
        valid.flags.writeable = False
        return valid

    @classmethod
    def sample_loci(
        cls,
        db: Database,
        n: int | None = None,
        *,
        seed: int | None = None,
    ) -> npt.NDArray[np.int32]:
        r"""Return all valid locus indices (`n=None`), a seeded random sample, or a stratified sample."""
        valid = cls.valid_locus_indices(db)
        if n is None or n >= len(valid):
            return valid
        k = max(1, n)
        if seed is not None:
            rng = np.random.default_rng(seed)
            sampled = np.sort(rng.choice(valid, size=k, replace=False)).astype(np.int32)
        else:
            picks = np.linspace(0, len(valid) - 1, num=k, dtype=np.int32)
            sampled = valid[np.unique(picks)].copy()
        sampled.flags.writeable = False
        return sampled

    def inject(
        self,
        db: Database,
        locus: int | str | None = None,
        *,
        strand: int | None = None,
        genome_id: str | None = None,
    ) -> InjectedGenome:
        r"""Select `locus` (randomly when `None`) from `db` and embed it into a synthetic `GenomeAssembly`."""
        rng = np.random.default_rng(self.seed)
        if locus is None:
            valid = self.valid_locus_indices(db)
            locus_idx = int(rng.choice(valid))
        elif isinstance(locus, str):
            locus_idx = db.loci.ids.index(locus)
        else:
            locus_idx = int(locus)

        chosen_strand = int(rng.choice([-1, 1])) if strand is None else (1 if strand >= 0 else -1)
        locus_rec = db.loci[locus_idx]
        locus_bytes = (
            locus_rec.seq.translate(BacterialTranslationTable._COMP)[::-1]
            if chosen_strand < 0
            else locus_rec.seq
        )

        contig_lengths = (self.flank_length, self.flank_length, *self.background_contig_lengths)
        bg_gen = SequenceGenerator(gc=self.gc, seed=int(rng.integers(1, 2**31 - 1)))
        bg_batch = bg_gen.generate(contig_lengths, prefix="bg")

        records: list[SeqRecord] = [
            SeqRecord(id="ctg_locus", seq=bg_batch[0].seq + locus_bytes + bg_batch[1].seq),
        ]
        for i, bg_len in enumerate(self.background_contig_lengths):
            if bg_len > 0:
                records.append(SeqRecord(id=f"ctg_bg_{i}", seq=bg_batch[2 + i].seq))

        asm_id = genome_id or f"sim_{db.metadata.keyword}_{locus_rec.id}"
        assembly = GenomeAssembly(id=asm_id, contigs=Sequences.from_records(records))
        return InjectedGenome(
            assembly=assembly,
            database_keyword=db.metadata.keyword,
            locus_idx=locus_idx,
            locus_name=locus_rec.id,
            strand=chosen_strand,
            expected_phenotype=db.serotypes[locus_idx],
        )


@dataclass(frozen=True, slots=True)
class SequenceShredder:
    r"""Vectorized and Numba-accelerated ragged batch sequence shredder and mutator."""

    n_fragments: int | None = None
    min_fragment_len: int = 2_000
    max_fragment_len: int = 12_000
    gap_len: int = 0
    flip_strand_prob: float = 0.5
    snp_rate: float = 0.0
    indel_rate: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        r"""Validate shredder configuration bounds."""
        if self.n_fragments is not None and self.n_fragments < 1:
            raise ValueError("n_fragments must be >= 1 when specified")
        if self.min_fragment_len < 1 or self.max_fragment_len < self.min_fragment_len:
            raise ValueError("Invalid fragment length bounds")
        if self.gap_len < 0:
            raise ValueError("gap_len must be >= 0")
        if not (0.0 <= self.flip_strand_prob <= 1.0):
            raise ValueError("flip_strand_prob must be in [0.0, 1.0]")
        if not (0.0 <= self.snp_rate <= 1.0) or not (0.0 <= self.indel_rate <= 1.0):
            raise ValueError("snp_rate and indel_rate must be in [0.0, 1.0]")

    @staticmethod
    @njit(parallel=True, cache=True, nogil=True)
    def _count_mutated_lengths_kernel(
        lengths: npt.NDArray[np.int32],
        indel_rate: float,
        seed: int,
    ) -> npt.NDArray[np.int32]:
        n_seqs = len(lengths)
        out_lengths = np.empty(n_seqs, dtype=np.int32)
        inv_53 = 1.0 / 9007199254740992.0
        golden = np.uint64(0x9E3779B97F4A7C15)
        mul1 = np.uint64(0xBF58476D1CE4E5B9)
        mul2 = np.uint64(0x94D049BB133111EB)
        base_seed = np.uint64(seed)

        for idx in prange(n_seqs):
            seq_len = int(lengths[idx])
            state = base_seed ^ (np.uint64(idx + 1) * golden)
            new_len = 0
            for _ in range(seq_len):
                state = state + golden
                z = state
                z = (z ^ (z >> np.uint64(30))) * mul1
                z = (z ^ (z >> np.uint64(27))) * mul2
                z = z ^ (z >> np.uint64(31))
                u_indel = float(z >> np.uint64(11)) * inv_53
                state = state + golden

                if indel_rate > 0.0 and u_indel < indel_rate:
                    if u_indel < indel_rate * 0.5:
                        continue
                    new_len += 2
                else:
                    new_len += 1
            out_lengths[idx] = new_len
        return out_lengths

    @staticmethod
    @njit(parallel=True, cache=True, nogil=True)
    def _apply_mutations_kernel(
        seqs: npt.NDArray[np.uint8],
        offsets: npt.NDArray[np.int32],
        lengths: npt.NDArray[np.int32],
        out_offsets: npt.NDArray[np.int32],
        out_lengths: npt.NDArray[np.int32],
        snp_rate: float,
        indel_rate: float,
        seed: int,
    ) -> npt.NDArray[np.uint8]:
        n_seqs = len(lengths)
        total_len = int(out_offsets[n_seqs - 1]) + int(out_lengths[n_seqs - 1]) if n_seqs > 0 else 0
        out = np.empty(total_len, dtype=np.uint8)
        if total_len == 0:
            return out

        bases = np.array([65, 67, 71, 84], dtype=np.uint8)
        inv_53 = 1.0 / 9007199254740992.0
        golden = np.uint64(0x9E3779B97F4A7C15)
        mul1 = np.uint64(0xBF58476D1CE4E5B9)
        mul2 = np.uint64(0x94D049BB133111EB)
        base_seed = np.uint64(seed)

        for idx in prange(n_seqs):
            seq_len = int(lengths[idx])
            in_start = int(offsets[idx])
            write_pos = int(out_offsets[idx])
            state = base_seed ^ (np.uint64(idx + 1) * golden)

            for j in range(seq_len):
                state = state + golden
                z1 = state
                z1 = (z1 ^ (z1 >> np.uint64(30))) * mul1
                z1 = (z1 ^ (z1 >> np.uint64(27))) * mul2
                z1 = z1 ^ (z1 >> np.uint64(31))
                u_indel = float(z1 >> np.uint64(11)) * inv_53

                state = state + golden
                z2 = state
                z2 = (z2 ^ (z2 >> np.uint64(30))) * mul1
                z2 = (z2 ^ (z2 >> np.uint64(27))) * mul2
                z2 = z2 ^ (z2 >> np.uint64(31))
                u_snp = float(z2 >> np.uint64(11)) * inv_53
                alt_pick = int(z2 & np.uint64(3))

                orig_b = seqs[in_start + j]

                if indel_rate > 0.0 and u_indel < indel_rate:
                    if u_indel < indel_rate * 0.5:
                        continue
                    out[write_pos] = bases[alt_pick]
                    out[write_pos + 1] = orig_b
                    write_pos += 2
                elif snp_rate > 0.0 and u_snp < snp_rate:
                    cur_idx = 0
                    for k in range(4):
                        if bases[k] == orig_b:
                            cur_idx = k
                            break
                    offset_choice = 1 + (int(z2 >> np.uint64(2)) % 3)
                    out[write_pos] = bases[(cur_idx + offset_choice) % 4]
                    write_pos += 1
                else:
                    out[write_pos] = orig_b
                    write_pos += 1
        return out

    def shred(
        self,
        seqs: Sequences,
        *,
        cut_positions: Sequence[Sequence[int]] | None = None,
    ) -> Sequences:
        r"""Shred a `Sequences` batch into fragmented, strand-flipped, and/or mutated contigs."""
        if len(seqs) == 0:
            return Sequences.empty()

        rng = np.random.default_rng(self.seed)
        parent_indices: list[int] = []
        starts: list[int] = []
        ends: list[int] = []
        new_ids: list[str] = []

        for seq_idx in range(len(seqs)):
            seq_len = int(seqs.lengths[seq_idx])
            if seq_len == 0:
                continue

            if cut_positions is not None:
                cuts = sorted(c for c in cut_positions[seq_idx] if 0 < c < seq_len)
            elif self.n_fragments is not None:
                if self.n_fragments == 1 or seq_len <= self.n_fragments:
                    cuts = []
                else:
                    ideal = np.linspace(0, seq_len, num=self.n_fragments + 1, dtype=np.int32)[1:-1]
                    max_jitter = max(1, seq_len // (self.n_fragments * 8))
                    jitter = rng.integers(-max_jitter, max_jitter + 1, size=len(ideal))
                    cuts = sorted({int(np.clip(c + j, 1, seq_len - 1)) for c, j in zip(ideal, jitter)})
            else:
                cuts = []
                pos = 0
                while seq_len - pos > self.max_fragment_len:
                    step = int(rng.integers(self.min_fragment_len, self.max_fragment_len + 1))
                    if pos + step >= seq_len:
                        break
                    cuts.append(pos + step)
                    pos += step + self.gap_len

            boundaries = [0, *cuts, seq_len]
            frag_num = 0
            for b_i in range(len(boundaries) - 1):
                s = boundaries[b_i] + (self.gap_len // 2 if b_i > 0 else 0)
                e = boundaries[b_i + 1] - (self.gap_len - self.gap_len // 2 if b_i < len(boundaries) - 2 else 0)
                if e > s:
                    parent_indices.append(seq_idx)
                    starts.append(s)
                    ends.append(e)
                    new_ids.append(f"{seqs.ids[seq_idx]}_frag{frag_num}")
                    frag_num += 1

        if not parent_indices:
            return Sequences.empty()

        n_out = len(parent_indices)
        flip_mask = rng.random(n_out) < self.flip_strand_prob
        strands = np.where(flip_mask, -1, 1).astype(np.int8)

        shredded = seqs.extract(
            np.array(parent_indices, dtype=np.int32),
            np.array(starts, dtype=np.int32),
            np.array(ends, dtype=np.int32),
            strands,
            new_ids=tuple(new_ids),
        )

        if self.snp_rate == 0.0 and self.indel_rate == 0.0:
            return shredded

        mut_seed = int(rng.integers(1, 2**31 - 1))
        if self.indel_rate > 0.0:
            out_lengths = self._count_mutated_lengths_kernel(shredded.lengths, float(self.indel_rate), mut_seed)
            out_lengths.flags.writeable = False
            out_offsets = SequenceGenerator.compute_offsets(out_lengths)
        else:
            out_lengths = shredded.lengths
            out_offsets = shredded.offsets

        mut_seqs = self._apply_mutations_kernel(
            shredded.seqs,
            shredded.offsets,
            shredded.lengths,
            out_offsets,
            out_lengths,
            float(self.snp_rate),
            float(self.indel_rate),
            mut_seed,
        )
        mut_seqs.flags.writeable = False
        return Sequences(ids=shredded.ids, seqs=mut_seqs, offsets=out_offsets, lengths=out_lengths)

    def shred_assembly(self, assembly: GenomeAssembly, *, suffix: str = "shredded") -> GenomeAssembly:
        r"""Shred all contigs in a `GenomeAssembly` and return a new `GenomeAssembly`."""
        return GenomeAssembly(id=f"{assembly.id}_{suffix}", contigs=self.shred(assembly.contigs))


class MaskedSerotyper(Serotyper):
    r"""`Serotyper` subclass that applies locus alignment masks for Leave-One-Locus-Out validation."""

    def __init__(
        self,
        db: Database,
        max_other_genes: int = 1,
        min_completeness: float = 0.5,
        allow_below_threshold: bool = False,
        preset: rammappy.Preset | None = None,
        scoring_metric: str = "scores",
        min_gene_coverage: float = 0.20,
        partial_edge_tolerance: int = 5,
    ) -> None:
        super().__init__(
            db=db,
            max_other_genes=max_other_genes,
            min_completeness=min_completeness,
            allow_below_threshold=allow_below_threshold,
            preset=preset,
            scoring_metric=scoring_metric,
            min_gene_coverage=min_gene_coverage,
            partial_edge_tolerance=partial_edge_tolerance,
        )
        self._locus_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(self._db.loci.ids)}

    @classmethod
    def ensure(cls, serotyper: Serotyper) -> MaskedSerotyper:
        r"""Return `serotyper` if already a `MaskedSerotyper`, else wrap its precomputed buffers in $O(1)$."""
        if isinstance(serotyper, cls):
            return serotyper
        inst = object.__new__(cls)
        inst._db = serotyper._db
        inst.max_other_genes = serotyper.max_other_genes
        inst.min_completeness = serotyper.min_completeness
        inst.allow_below_threshold = serotyper.allow_below_threshold
        inst.preset = serotyper.preset
        inst.scoring_metric = serotyper.scoring_metric
        inst.min_gene_coverage = serotyper.min_gene_coverage
        inst.partial_edge_tolerance = serotyper.partial_edge_tolerance
        inst._protein_aligner = serotyper._protein_aligner
        inst._extra_locus_mask = serotyper._extra_locus_mask
        inst._expected_genes_per_locus = serotyper._expected_genes_per_locus
        inst._gene_seqs = serotyper._gene_seqs
        inst._gene_meta = serotyper._gene_meta
        inst._locus_name_to_idx = {name: i for i, name in enumerate(serotyper._db.loci.ids)}
        return inst

    def resolve_exclude_mask(self, exclude_loci: LocusFilter | None) -> npt.NDArray[np.bool_]:
        r"""Normalize an optional locus exclusion filter into a read-only 1D boolean mask over `db.loci`."""
        n_loci = len(self._db.loci)
        mask = np.zeros(n_loci, dtype=bool)
        if exclude_loci is not None:
            if isinstance(exclude_loci, np.ndarray):
                if exclude_loci.dtype != bool or exclude_loci.shape != (n_loci,):
                    raise ValueError(f"Boolean exclude_loci mask must have dtype=bool and shape ({n_loci},)")
                mask = exclude_loci.copy()
            else:
                items: Sequence[int | str]
                if isinstance(exclude_loci, str):
                    items = (exclude_loci,)
                elif isinstance(exclude_loci, (int, np.integer)):
                    items = (int(exclude_loci),)
                else:
                    items = exclude_loci

                for item in items:
                    if isinstance(item, str):
                        if item not in self._locus_name_to_idx:
                            raise KeyError(f"Unknown locus {item!r} in database {self._db.metadata.keyword!r}")
                        mask[self._locus_name_to_idx[item]] = True
                    else:
                        idx = int(item)
                        if idx < 0 or idx >= n_loci:
                            raise IndexError(f"Locus index {idx} out of range for {n_loci} loci")
                        mask[idx] = True
        mask.flags.writeable = False
        return mask

    def score_masked(
        self,
        genome: GenomeAssembly,
        gene_alns: Alignments,
        *,
        exclude_loci: LocusFilter | None = None,
    ) -> SerotypingResult | None:
        r"""Filter `gene_alns` and score candidate loci with `exclude_loci` masked out."""
        exclude_mask = self.resolve_exclude_mask(exclude_loci)
        if not bool(np.any(exclude_mask)):
            return self.score_and_reconstruct(genome, gene_alns)

        disallowed = self._extra_locus_mask | exclude_mask
        disallowed.flags.writeable = False
        if bool(np.all(disallowed)):
            return None

        if len(gene_alns) > 0:
            raw_q_indices = gene_alns.q_names.astype(np.int32)
            keep_alns = (~exclude_mask[self._db.gene_locus_indices[raw_q_indices]]) | self._db.extra_genes[
                raw_q_indices
            ]
            gene_alns = gene_alns[keep_alns]

        view = copy.copy(self)
        view._extra_locus_mask = disallowed
        return view.score_and_reconstruct(genome, gene_alns)

    def __call__(
        self,
        genome: GenomeAssembly | str | Path,
        *,
        exclude_loci: LocusFilter | None = None,
    ) -> SerotypingResult | None:
        r"""Align `genome` once and score with optional `exclude_loci` mask."""
        genome_obj = GenomeAssembly.ensure(genome)
        return self.score_masked(genome_obj, self.align_genes(genome_obj), exclude_loci=exclude_loci)


@dataclass(frozen=True, slots=True)
class LoloFoldResult:
    r"""Immutable result of single-pass positive self-typing and Leave-One-Locus-Out evaluation."""

    locus_idx: int
    locus_name: str
    self_result: SerotypingResult
    lolo_result: SerotypingResult | None


@dataclass(frozen=True, slots=True)
class LocusCrossValidator:
    r"""Single-pass self-consistency and Leave-One-Locus-Out (LOLO) cross-validator.

    Reuses a single `Database` and `MaskedSerotyper` instance across all evaluated loci and executes
    `align_genes` once per locus to evaluate both:
        1. Unmasked positive self-typing (`exclude_loci=None`).
        2. Masked Leave-One-Locus-Out (`exclude_loci=locus_idx`): two loci are deemed
           indistinguishable if holding one out yields `typeable=True` with `problems == NONE`
           and all inside genes `>= db.metadata.id_threshold`, unless declared in `[lolo_allowlist]`.
    """

    allowlist: tuple[tuple[str, tuple[str, ...]], ...] = ()
    flank_length: int = 2_000
    gc: float = 0.5
    seed: int = 42

    def is_allowed_cross_hit(self, held_out_locus: str, runner_up_locus: str) -> bool:
        r"""Check whether `(held_out_locus, runner_up_locus)` matches any wildcard rule in `allowlist`."""
        for query_pat, target_pats in self.allowlist:
            if fnmatchcase(held_out_locus, query_pat):
                if any(fnmatchcase(runner_up_locus, t_pat) for t_pat in target_pats):
                    return True
        return False

    @staticmethod
    def is_deemed_identical(result: SerotypingResult, id_threshold: float) -> bool:
        r"""Return `True` if `result` is typeable with no problems and all inside genes meet `id_threshold`."""
        if not result.typeable or result.problems != SerotypingProblem.NONE:
            return False
        inside_mask = result.gene_hits.is_inside
        if not bool(np.any(inside_mask)):
            return False
        return bool(np.all(result.protein_identities[inside_mask] >= id_threshold))

    def evaluate_locus(
        self,
        db: Database,
        serotyper: Serotyper,
        locus_idx: int,
    ) -> LoloFoldResult:
        r"""Run one `align_genes` pass on `locus_idx` and compute both self-typing and LOLO calls."""
        masked = MaskedSerotyper.ensure(serotyper)
        injected = LocusInjector(
            gc=self.gc,
            flank_length=self.flank_length,
            background_contig_lengths=(),
            seed=self.seed + locus_idx,
        ).inject(db, locus=locus_idx, strand=1)

        gene_alns = masked.align_genes(injected.assembly)
        self_res = masked.score_and_reconstruct(injected.assembly, gene_alns)
        assert self_res is not None, f"Self-typing returned None for {injected.locus_name}"
        lolo_res = masked.score_masked(injected.assembly, gene_alns, exclude_loci=locus_idx)
        return LoloFoldResult(
            locus_idx=locus_idx,
            locus_name=injected.locus_name,
            self_result=self_res,
            lolo_result=lolo_res,
        )

    def verify_fold(self, fold: LoloFoldResult, *, id_threshold: float) -> None:
        r"""Assert positive self-typing and verify held-out locus is not indistinguishable from runner-up."""
        sr = fold.self_result
        assert sr.best_locus_name == fold.locus_name, (
            f"Self-typing locus mismatch for {fold.locus_name}: got {sr.best_locus_name}"
        )
        assert sr.best_locus_completeness == 1.0, (
            f"Self-typing completeness < 100% for {fold.locus_name}: {sr.best_locus_completeness:.2%}"
        )
        assert sr.typeable is True, (
            f"Self-typing failed typeability for {fold.locus_name} "
            f"(problems={sr.problems.to_symbols().decode()!r})"
        )
        assert sr.problems == SerotypingProblem.NONE, (
            f"Unexpected self-typing problems on {fold.locus_name}: {sr.problems!r}"
        )

        lr = fold.lolo_result
        if lr is None:
            return
        assert lr.best_locus_name != fold.locus_name, (
            f"LOLO exclusion failed: held-out locus {fold.locus_name} was still selected!"
        )
        if self.is_deemed_identical(lr, id_threshold) and not self.is_allowed_cross_hit(
            fold.locus_name, lr.best_locus_name
        ):
            min_aa_id = float(np.min(lr.protein_identities[lr.gene_hits.is_inside]))
            raise AssertionError(
                f"LOLO indistinguishable locus pair when holding out {fold.locus_name!r}: "
                f"runner-up {lr.best_locus_name!r} called with zero problems and all genes >= "
                f"id_threshold ({id_threshold:.2f}%, min_gene_aa_id={min_aa_id:.2f}%, "
                f"mean_aa_id={lr.percent_identity:.2f}%, phenotype={lr.phenotype!r}). "
                f"If {fold.locus_name!r} and {lr.best_locus_name!r} are intentional allelic sub-loci, "
                f"add them to [lolo_allowlist]."
            )

    def validate(
        self,
        db: Database,
        serotyper: Serotyper | None = None,
        *,
        locus_indices: Sequence[int] | npt.NDArray[np.integer] | None = None,
    ) -> tuple[LoloFoldResult, ...]:
        r"""Run and verify single-pass self-typing + LOLO across `locus_indices` (all valid loci if `None`)."""
        engine = serotyper if serotyper is not None else Serotyper(db)
        indices = LocusInjector.valid_locus_indices(db) if locus_indices is None else locus_indices
        threshold = float(db.metadata.id_threshold)
        results: list[LoloFoldResult] = []
        errors: list[str] = []
        for idx in indices:
            fold = self.evaluate_locus(db, engine, int(idx))
            results.append(fold)
            try:
                self.verify_fold(fold, id_threshold=threshold)
            except AssertionError as exc:
                errors.append(str(exc))
        if errors:
            raise AssertionError(
                f"{len(errors)} validation failure(s) in database {db.metadata.keyword!r}:\n  - "
                + "\n  - ".join(errors)
            )
        return tuple(results)
