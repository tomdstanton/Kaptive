r"""Class-first NumPy/Numba simulation, controllable degradation, and LOLO validation toolkit.

Follows the SeroVerse zero-copy Structure-of-Arrays (SoA) and strict class-first paradigms:
all random sequence generation, full-chromosome locus injection, locus-aware draft shredding,
Stage-1 orthogonal degradation ladders, and Stage-2 single-pass Leave-One-Locus-Out (LOLO)
evaluation live on immutable classes backed by stateless SplitMix64 Numba kernels with zero
module-level functions.
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
    r"""Immutable record pairing a synthetic `GenomeAssembly` with exact locus ground-truth coordinates."""

    assembly: GenomeAssembly
    database_keyword: str
    locus_idx: int
    locus_name: str
    strand: int
    expected_phenotype: str
    contig_idx: int
    locus_start: int
    locus_end: int
    gc: float


@dataclass(frozen=True, slots=True)
class LocusInjector:
    r"""Selects reference loci from a `Database` and splices them into full-size bacterial genomes.

    Attributes:
        gc (float | None): Target background GC fraction, or `None` to sample uniformly from
            `[gc_min, gc_max]`. Defaults to `0.5`.
        gc_min (float): Lower bound when sampling random GC (`gc=None`). Defaults to `0.25`.
        gc_max (float): Upper bound when sampling random GC (`gc=None`). Defaults to `0.75`.
        genome_size (int): Target chromosome size in base pairs. Defaults to `5_000_000` (`5 Mbp`).
        min_flank (int): Minimum flanking sequence preserved on both sides of the spliced locus.
            Defaults to `2_000`.
        plasmid_lengths (tuple[int, ...]): Optional unlinked plasmid contig lengths. Defaults to `()`.
        seed (int): Deterministic PRNG seed. Defaults to `42`.
    """

    gc: float | None = 0.5
    gc_min: float = 0.25
    gc_max: float = 0.75
    genome_size: int = 5_000_000
    min_flank: int = 2_000
    plasmid_lengths: tuple[int, ...] = ()
    seed: int = 42

    def __post_init__(self) -> None:
        r"""Validate GC and genome size bounds."""
        if self.gc is not None and not (0.0 <= self.gc <= 1.0):
            raise ValueError(f"gc must be in [0.0, 1.0] or None, got {self.gc}")
        if not (0.0 <= self.gc_min <= self.gc_max <= 1.0):
            raise ValueError(f"Invalid GC sampling range [{self.gc_min}, {self.gc_max}]")
        if self.genome_size < 0 or self.min_flank < 0:
            raise ValueError("genome_size and min_flank must be non-negative")

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
        r"""Select `locus` (randomly when `None`) and splice it at a random offset in a chromosome."""
        rng = np.random.default_rng(self.seed)
        if locus is None:
            valid = self.valid_locus_indices(db)
            locus_idx = int(rng.choice(valid))
        elif isinstance(locus, str):
            locus_idx = db.loci.ids.index(locus)
        else:
            locus_idx = int(locus)

        chosen_strand = int(rng.choice([-1, 1])) if strand is None else (1 if strand >= 0 else -1)
        chosen_gc = float(rng.uniform(self.gc_min, self.gc_max)) if self.gc is None else float(self.gc)

        locus_rec = db.loci[locus_idx]
        locus_bytes = (
            locus_rec.seq.translate(BacterialTranslationTable._COMP)[::-1]
            if chosen_strand < 0
            else locus_rec.seq
        )
        locus_len = len(locus_bytes)
        bg_total = max(2 * self.min_flank, self.genome_size - locus_len)
        if bg_total == 2 * self.min_flank:
            left_len = self.min_flank
        else:
            left_len = int(rng.integers(self.min_flank, bg_total - self.min_flank + 1))
        right_len = bg_total - left_len

        contig_lengths = (left_len, right_len, *self.plasmid_lengths)
        bg_gen = SequenceGenerator(gc=chosen_gc, seed=int(rng.integers(1, 2**31 - 1)))
        bg_batch = bg_gen.generate(contig_lengths, prefix="bg")

        records: list[SeqRecord] = [
            SeqRecord(id="chr_0", seq=bg_batch[0].seq + locus_bytes + bg_batch[1].seq),
        ]
        for i, pl_len in enumerate(self.plasmid_lengths):
            if pl_len > 0:
                records.append(SeqRecord(id=f"plasmid_{i}", seq=bg_batch[2 + i].seq))

        asm_id = genome_id or f"sim_{db.metadata.keyword}_{locus_rec.id}"
        assembly = GenomeAssembly(id=asm_id, contigs=Sequences.from_records(records))
        return InjectedGenome(
            assembly=assembly,
            database_keyword=db.metadata.keyword,
            locus_idx=locus_idx,
            locus_name=locus_rec.id,
            strand=chosen_strand,
            expected_phenotype=db.serotypes[locus_idx],
            contig_idx=0,
            locus_start=left_len,
            locus_end=left_len + locus_len,
            gc=chosen_gc,
        )


@dataclass(frozen=True, slots=True)
class SequenceShredder:
    r"""Vectorized and Numba-accelerated ragged batch sequence shredder and locus-aware mutator."""

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

    @staticmethod
    def _freeze_nonempty_sequences(batch: Sequences) -> Sequences:
        r"""Drop zero-length records and lock all internal NumPy arrays to read-only."""
        if len(batch) == 0:
            return Sequences.empty()
        if bool(np.any(batch.lengths <= 0)):
            keep = batch.lengths > 0
            if not bool(np.any(keep)):
                return Sequences.empty()
            batch = batch[keep]
        batch.seqs.flags.writeable = False
        batch.offsets.flags.writeable = False
        batch.lengths.flags.writeable = False
        return batch

    def _mutate_batch(self, batch: Sequences, mut_seed: int) -> Sequences:
        r"""Apply SNP and indel kernels to `batch`, returning a new read-only `Sequences` batch."""
        if len(batch) == 0 or (self.snp_rate == 0.0 and self.indel_rate == 0.0):
            return self._freeze_nonempty_sequences(batch)
        if self.indel_rate > 0.0:
            out_lengths = self._count_mutated_lengths_kernel(batch.lengths, float(self.indel_rate), mut_seed)
            out_lengths.flags.writeable = False
            out_offsets = SequenceGenerator.compute_offsets(out_lengths)
        else:
            out_lengths = batch.lengths
            out_offsets = batch.offsets

        mut_seqs = self._apply_mutations_kernel(
            batch.seqs,
            batch.offsets,
            batch.lengths,
            out_offsets,
            out_lengths,
            float(self.snp_rate),
            float(self.indel_rate),
            mut_seed,
        )
        return self._freeze_nonempty_sequences(
            Sequences(ids=batch.ids, seqs=mut_seqs, offsets=out_offsets, lengths=out_lengths)
        )

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
        return self._mutate_batch(shredded, int(rng.integers(1, 2**31 - 1)))

    def shred_assembly(self, assembly: GenomeAssembly, *, suffix: str = "shredded") -> GenomeAssembly:
        r"""Shred all contigs in a `GenomeAssembly` and return a new `GenomeAssembly`."""
        return GenomeAssembly(id=f"{assembly.id}_{suffix}", contigs=self.shred(assembly.contigs))

    @staticmethod
    def _mapped_expected_gene_intervals(
        db: Database,
        injected: InjectedGenome,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        r"""Map expected gene coordinates of `injected.locus_idx` onto `injected` chromosome coordinates."""
        off = int(db.locus_gene_offsets[injected.locus_idx])
        length = int(db.locus_gene_lengths[injected.locus_idx])
        not_extra = ~db.extra_genes[off : off + length]
        raw_s = db.gene_intervals.starts[off : off + length][not_extra].astype(np.int32)
        raw_e = db.gene_intervals.ends[off : off + length][not_extra].astype(np.int32)
        if injected.strand >= 0:
            chr_s = injected.locus_start + raw_s
            chr_e = injected.locus_start + raw_e
        else:
            chr_s = injected.locus_end - raw_e
            chr_e = injected.locus_end - raw_s
        order = np.argsort(chr_s)
        return chr_s[order], chr_e[order]

    @staticmethod
    def _sample_span_cuts(
        span_start: int,
        span_end: int,
        target_len: int,
        rng: np.random.Generator,
    ) -> list[int]:
        r"""Sample jittered draft contig cut coordinates strictly inside `(span_start, span_end)`."""
        cuts: list[int] = []
        low_step = max(500, int(target_len * 0.75))
        high_step = max(low_step + 1, int(target_len * 1.25))
        pos = span_start
        while span_end - pos > high_step:
            step = int(rng.integers(low_step, high_step + 1))
            if pos + step >= span_end - 500:
                break
            cuts.append(pos + step)
            pos += step
        return cuts

    def shred_injected(
        self,
        injected: InjectedGenome,
        *,
        db: Database | None = None,
        draft_contig_len: int = 50_000,
        locus_cuts: int = 0,
        locus_dropout_frac: float = 0.0,
        suffix: str = "degraded",
    ) -> GenomeAssembly:
        r"""Fragment a full-size `InjectedGenome` while applying controlled locus cuts, dropout, and mutations."""
        if locus_cuts < 0:
            raise ValueError("locus_cuts must be >= 0")
        if not (0.0 <= locus_dropout_frac <= 1.0):
            raise ValueError("locus_dropout_frac must be in [0.0, 1.0]")

        rng = np.random.default_rng(self.seed)
        contigs = injected.assembly.contigs
        chr_bytes = contigs[injected.contig_idx].seq
        chr_len = len(chr_bytes)
        loc_s = injected.locus_start
        loc_e = injected.locus_end

        g_starts: npt.NDArray[np.int32] | None = None
        g_ends: npt.NDArray[np.int32] | None = None
        if db is not None:
            g_starts, g_ends = self._mapped_expected_gene_intervals(db, injected)

        # 1. Resolve contiguous dropout interval `[del_start, loc_e]` inside the locus
        del_start = loc_e
        if locus_dropout_frac > 0.0:
            if g_starts is not None and g_ends is not None and len(g_starts) >= 2:
                n_genes = len(g_starts)
                raw_drop = int(round(n_genes * locus_dropout_frac))
                if locus_dropout_frac < 0.5:
                    n_drop = max(1, min(n_genes // 2, raw_drop))
                else:
                    n_drop = max((n_genes // 2) + 1, min(n_genes, raw_drop))
                del_start = loc_s if n_drop >= n_genes else int(g_ends[n_genes - n_drop - 1])
                g_starts = g_starts[: n_genes - n_drop]
                g_ends = g_ends[: n_genes - n_drop]
            else:
                del_len = int(round((loc_e - loc_s) * locus_dropout_frac))
                del_start = max(loc_s, loc_e - del_len)

        # 2. Apply controlled SNP/indel divergence specifically inside the surviving locus region
        surviving_locus = chr_bytes[loc_s:del_start]
        if len(surviving_locus) > 0 and (self.snp_rate > 0.0 or self.indel_rate > 0.0):
            locus_batch = Sequences.from_records([SeqRecord(id="locus", seq=surviving_locus)])
            mut_locus = self._mutate_batch(locus_batch, int(rng.integers(1, 2**31 - 1)))
            surviving_locus = mut_locus[0].seq if len(mut_locus) > 0 else b""
            g_starts = None
            g_ends = None

        surv_end = loc_s + len(surviving_locus)

        # 3. Build chromosome arms (split at any excised dropout gap) and sample draft + locus cuts
        arm_records: list[SeqRecord] = []
        arm_cuts: list[list[int]] = []

        left_arm = chr_bytes[:loc_s] + surviving_locus + (chr_bytes[loc_e:] if del_start == loc_e else b"")
        left_cuts = self._sample_span_cuts(0, max(0, loc_s - 500), draft_contig_len, rng)

        if locus_cuts > 0 and len(surviving_locus) > locus_cuts + 1:
            if g_starts is not None and g_ends is not None and len(g_starts) > locus_cuts:
                n_surv = len(g_starts)
                for c_i in range(locus_cuts):
                    split_idx = int(round((c_i + 1) * n_surv / (locus_cuts + 1)))
                    split_idx = max(1, min(n_surv - 1, split_idx))
                    cut_coord = int((int(g_ends[split_idx - 1]) + int(g_starts[split_idx])) // 2)
                    if loc_s < cut_coord < surv_end:
                        left_cuts.append(cut_coord)
            else:
                even_cuts = np.linspace(loc_s, surv_end, num=locus_cuts + 2, dtype=np.int32)[1:-1]
                left_cuts.extend(int(c) for c in even_cuts if loc_s < int(c) < surv_end)

        if del_start == loc_e:
            right_offset = surv_end + 500
            if right_offset < len(left_arm):
                left_cuts.extend(self._sample_span_cuts(right_offset, len(left_arm), draft_contig_len, rng))
            arm_records.append(SeqRecord(id="chr_0", seq=left_arm))
            arm_cuts.append(sorted(set(left_cuts)))
        else:
            if len(left_arm) > 0:
                arm_records.append(SeqRecord(id="chr_0a", seq=left_arm))
                arm_cuts.append(sorted(set(left_cuts)))
            right_arm = chr_bytes[loc_e:]
            if len(right_arm) > 0:
                arm_records.append(SeqRecord(id="chr_0b", seq=right_arm))
                arm_cuts.append(self._sample_span_cuts(500, len(right_arm), draft_contig_len, rng))

        for idx in range(len(contigs)):
            if idx != injected.contig_idx and int(contigs.lengths[idx]) > 0:
                rec = contigs[idx]
                arm_records.append(rec)
                arm_cuts.append(self._sample_span_cuts(0, len(rec.seq), draft_contig_len, rng))

        arm_batch = Sequences.from_records(arm_records)
        cutter = SequenceShredder(
            gap_len=self.gap_len,
            flip_strand_prob=self.flip_strand_prob,
            snp_rate=0.0,
            indel_rate=0.0,
            seed=int(rng.integers(1, 2**31 - 1)),
        )
        draft_contigs = cutter.shred(arm_batch, cut_positions=arm_cuts)
        if len(draft_contigs) > 1:
            perm = rng.permutation(len(draft_contigs))
            draft_contigs = self._freeze_nonempty_sequences(draft_contigs[perm])

        return GenomeAssembly(id=f"{injected.assembly.id}_{suffix}", contigs=draft_contigs)


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
class Stage1LocusReport:
    r"""Immutable record of Stage-1 full-chromosome GC and orthogonal degradation results for one locus."""

    locus_idx: int
    locus_name: str
    gc: float
    strand: int
    intact_result: SerotypingResult
    fragmentation_results: tuple[SerotypingResult, ...]
    moderate_dropout_result: SerotypingResult
    severe_dropout_result: SerotypingResult
    moderate_divergence_result: SerotypingResult
    severe_divergence_result: SerotypingResult


@dataclass(frozen=True, slots=True)
class Stage1DegradationValidator:
    r"""Stage-1 validator: 5 Mbp chromosome GC control + 3 orthogonal assembly destruction ladders.

    Evaluates each sampled locus across:
        0. **Intact baseline**: Full `genome_size` chromosome at target `gc` (`0.25 / 0.50 / 0.75`).
        1. **Fragmentation axis**: `fragmentation_cuts` (`2`, `4`) inside the locus + draft background.
        2. **Dropout axis**: `moderate_dropout` (`25%`) -> `severe_dropout` (`75%` -> `typeable=False`).
        3. **Divergence axis**: `moderate_snp_rate` (`1%`) -> `severe_snp_rate` (`35%` + indels -> `typeable=False`).
    """

    genome_size: int = 5_000_000
    draft_contig_len: int = 50_000
    gc_levels: tuple[float, ...] = (0.25, 0.50, 0.75)
    fragmentation_cuts: tuple[int, ...] = (2, 4)
    moderate_dropout: float = 0.25
    severe_dropout: float = 0.75
    moderate_snp_rate: float = 0.01
    severe_snp_rate: float = 0.35
    severe_indel_rate: float = 0.05
    seed: int = 42

    def evaluate_locus(
        self,
        db: Database,
        serotyper: Serotyper,
        locus_idx: int,
        *,
        gc: float,
    ) -> Stage1LocusReport:
        r"""Inject `locus_idx` into a `genome_size` chromosome at `gc` and run all 3 orthogonal ladders."""
        base_seed = self.seed + locus_idx * 17
        injected = LocusInjector(
            gc=gc,
            genome_size=self.genome_size,
            seed=base_seed,
        ).inject(db, locus=locus_idx)

        intact_res = serotyper(injected.assembly)
        assert intact_res is not None

        frag_results: list[SerotypingResult] = []
        for step_i, n_cuts in enumerate(self.fragmentation_cuts):
            frag_asm = SequenceShredder(
                flip_strand_prob=0.5,
                seed=base_seed + 1 + step_i,
            ).shred_injected(
                injected,
                db=db,
                draft_contig_len=self.draft_contig_len,
                locus_cuts=n_cuts,
                suffix=f"frag{n_cuts}",
            )
            f_res = serotyper(frag_asm)
            assert f_res is not None
            frag_results.append(f_res)

        mod_drop_asm = SequenceShredder(
            flip_strand_prob=0.5,
            seed=base_seed + 10,
        ).shred_injected(
            injected,
            db=db,
            draft_contig_len=self.draft_contig_len,
            locus_dropout_frac=self.moderate_dropout,
            suffix="drop25",
        )
        mod_drop_res = serotyper(mod_drop_asm)
        assert mod_drop_res is not None

        sev_drop_asm = SequenceShredder(
            flip_strand_prob=0.5,
            seed=base_seed + 11,
        ).shred_injected(
            injected,
            db=db,
            draft_contig_len=self.draft_contig_len,
            locus_dropout_frac=self.severe_dropout,
            suffix="drop75",
        )
        sev_drop_res = serotyper(sev_drop_asm)
        assert sev_drop_res is not None

        mod_div_asm = SequenceShredder(
            flip_strand_prob=0.5,
            snp_rate=self.moderate_snp_rate,
            indel_rate=0.0,
            seed=base_seed + 12,
        ).shred_injected(
            injected,
            db=db,
            draft_contig_len=self.draft_contig_len,
            suffix="div1",
        )
        mod_div_res = serotyper(mod_div_asm)
        assert mod_div_res is not None

        sev_div_asm = SequenceShredder(
            flip_strand_prob=0.5,
            snp_rate=self.severe_snp_rate,
            indel_rate=self.severe_indel_rate,
            seed=base_seed + 13,
        ).shred_injected(
            injected,
            db=db,
            draft_contig_len=self.draft_contig_len,
            suffix="div35",
        )
        sev_div_res = serotyper(sev_div_asm)
        assert sev_div_res is not None

        return Stage1LocusReport(
            locus_idx=locus_idx,
            locus_name=injected.locus_name,
            gc=injected.gc,
            strand=injected.strand,
            intact_result=intact_res,
            fragmentation_results=tuple(frag_results),
            moderate_dropout_result=mod_drop_res,
            severe_dropout_result=sev_drop_res,
            moderate_divergence_result=mod_div_res,
            severe_divergence_result=sev_div_res,
        )

    def verify_report(self, report: Stage1LocusReport) -> None:
        r"""Assert Stage-1 ground-truth invariants across intact and orthogonal degradation steps."""
        loc = report.locus_name
        ctx = f"[{loc}, gc={report.gc:.2f}, strand={report.strand:+d}]"

        ir = report.intact_result
        assert ir.best_locus_name == loc, f"{ctx} Intact mismatch: got {ir.best_locus_name!r}"
        assert ir.typeable is True, (
            f"{ctx} Intact untypeable (problems={ir.problems.to_symbols().decode()!r})"
        )

        for n_cuts, fr in zip(self.fragmentation_cuts, report.fragmentation_results):
            assert fr.best_locus_name == loc, f"{ctx} Frag({n_cuts} cuts) mismatch: got {fr.best_locus_name!r}"
            assert fr.typeable is True, f"{ctx} Frag({n_cuts} cuts) untypeable"
            assert len(fr.locus_pieces) >= 2, (
                f"{ctx} Frag({n_cuts} cuts) expected >=2 pieces, got {len(fr.locus_pieces)}"
            )
            assert bool(fr.problems & SerotypingProblem.FRAGMENTED), (
                f"{ctx} Frag({n_cuts} cuts) missing FRAGMENTED problem flag"
            )

        md = report.moderate_dropout_result
        assert md.best_locus_name == loc, f"{ctx} Dropout({self.moderate_dropout:.0%}) mismatch: got {md.best_locus_name!r}"
        assert md.typeable is True, f"{ctx} Dropout({self.moderate_dropout:.0%}) untypeable"
        assert 0.5 <= md.best_locus_completeness < 1.0, (
            f"{ctx} Dropout({self.moderate_dropout:.0%}) completeness={md.best_locus_completeness:.2%} "
            f"outside [50%, 100%)"
        )

        sd = report.severe_dropout_result
        assert sd.typeable is False, (
            f"{ctx} Dropout({self.severe_dropout:.0%}) stayed typeable "
            f"(called={sd.best_locus_name!r}, completeness={sd.best_locus_completeness:.2%})"
        )

        mv = report.moderate_divergence_result
        assert mv.best_locus_name == loc, (
            f"{ctx} Divergence({self.moderate_snp_rate:.1%}) mismatch: got {mv.best_locus_name!r}"
        )
        assert mv.typeable is True, (
            f"{ctx} Divergence({self.moderate_snp_rate:.1%}) untypeable "
            f"(problems={mv.problems.to_symbols().decode()!r})"
        )
        assert mv.percent_identity < 100.0, (
            f"{ctx} Divergence({self.moderate_snp_rate:.1%}) identity remained 100.0%"
        )

        sv = report.severe_divergence_result
        assert sv.typeable is False, (
            f"{ctx} Divergence({self.severe_snp_rate:.0%}) stayed typeable "
            f"(called={sv.best_locus_name!r}, completeness={sv.best_locus_completeness:.2%})"
        )

    def validate(
        self,
        db: Database,
        serotyper: Serotyper | None = None,
        *,
        locus_indices: Sequence[int] | npt.NDArray[np.integer] | None = None,
    ) -> tuple[Stage1LocusReport, ...]:
        r"""Run Stage-1 validation across `locus_indices`, cycling through `gc_levels`."""
        engine = serotyper if serotyper is not None else Serotyper(db)
        indices = LocusInjector.valid_locus_indices(db) if locus_indices is None else locus_indices
        reports: list[Stage1LocusReport] = []
        errors: list[str] = []
        for pos, raw_idx in enumerate(indices):
            gc_val = self.gc_levels[pos % len(self.gc_levels)]
            rep = self.evaluate_locus(db, engine, int(raw_idx), gc=gc_val)
            reports.append(rep)
            try:
                self.verify_report(rep)
            except AssertionError as exc:
                errors.append(str(exc))
        if errors:
            raise AssertionError(
                f"{len(errors)} Stage-1 degradation failure(s) in database {db.metadata.keyword!r}:\n  - "
                + "\n  - ".join(errors)
            )
        return tuple(reports)


@dataclass(frozen=True, slots=True)
class LoloFoldResult:
    r"""Immutable result of single-pass positive self-typing and Leave-One-Locus-Out evaluation."""

    locus_idx: int
    locus_name: str
    self_result: SerotypingResult
    lolo_result: SerotypingResult | None


@dataclass(frozen=True, slots=True)
class LocusCrossValidator:
    r"""Stage-2 single-pass self-consistency and Leave-One-Locus-Out (LOLO) cross-validator.

    Reuses a single `Database` and `MaskedSerotyper` instance across all evaluated loci and executes
    `align_genes` once per locus to evaluate both:
        1. Unmasked positive self-typing (`exclude_loci=None`).
        2. Masked Leave-One-Locus-Out (`exclude_loci=locus_idx`): two loci are deemed
           indistinguishable if holding one out yields `typeable=True` with `problems == NONE`
           and all inside genes `>= db.metadata.id_threshold`, unless declared in `[lolo_allowlist]`.
    """

    allowlist: tuple[tuple[str, tuple[str, ...]], ...] = ()
    genome_size: int = 25_000
    min_flank: int = 2_000
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
            genome_size=self.genome_size,
            min_flank=self.min_flank,
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
