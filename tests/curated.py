r"""Class-first decentralized `test.tsv` & `[lolo_allowlist]` loader and NCBI Datasets v2 FASTA cache."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import re
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, final

from kaptive.core.genome import GenomeAssembly
from kaptive.db.manager import DatabaseManager
from kaptive.serotyping.models import SerotypingProblem, SerotypingResult


@dataclass(frozen=True, slots=True)
class SerotypeExpectation:
    r"""Immutable declarative expectation parsed from a single `test.tsv` row."""

    _SYMBOL_MAP: ClassVar[tuple[tuple[str, SerotypingProblem], ...]] = (
        ("?", SerotypingProblem.FRAGMENTED),
        ("+", SerotypingProblem.UNEXPECTED_GENES),
        ("-", SerotypingProblem.MISSING_GENES),
        ("*", SerotypingProblem.NOVEL_GENES),
        ("!", SerotypingProblem.TRUNCATED_GENES),
    )

    source: str
    base_dir: Path | None
    remote_base_url: str | None
    genome: str
    database: str
    locus: str
    phenotype: str
    typeable: bool
    problems: SerotypingProblem | None = None
    missing_genes: tuple[str, ...] | None = None
    min_coverage: float | None = None
    min_identity: float | None = None
    sha256: str | None = None
    notes: str = ""

    @property
    def test_id(self) -> str:
        r"""Concise pytest node ID for parameterized test reporting."""
        return f"{self.database}-{self.genome}-{self.locus}"

    @classmethod
    def parse_bool(cls, raw: str) -> bool:
        r"""Parse boolean string tokens from `test.tsv`."""
        norm = raw.strip().lower()
        if norm in ("true", "t", "1", "yes", "y", "typeable"):
            return True
        if norm in ("false", "f", "0", "no", "n", "untypeable"):
            return False
        raise ValueError(f"Invalid boolean token in test.tsv: {raw!r}")

    @classmethod
    def parse_problems(cls, raw: str) -> SerotypingProblem | None:
        r"""Parse Kaptive problem symbols (`?-*+!`) or `|`-delimited flag names."""
        s = raw.strip()
        if not s:
            return None
        if s.lower() in ("none", "0", "ok"):
            return SerotypingProblem.NONE
        sym_lookup = dict(cls._SYMBOL_MAP)
        if all(ch in sym_lookup for ch in s):
            flags = SerotypingProblem.NONE
            for ch in s:
                flags |= sym_lookup[ch]
            return flags
        flags = SerotypingProblem.NONE
        for part in s.split("|"):
            name = part.strip().upper()
            if name and name != "NONE":
                flags |= SerotypingProblem[name]
        return flags

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, str | None],
        *,
        source: str,
        base_dir: Path | None = None,
        remote_base_url: str | None = None,
    ) -> SerotypeExpectation:
        r"""Construct an immutable `SerotypeExpectation` from a `csv.DictReader` row."""
        missing_raw = (row.get("missing_genes") or "").strip()
        missing: tuple[str, ...] | None
        if not missing_raw:
            missing = None
        elif missing_raw.lower() == "none":
            missing = ()
        else:
            missing = tuple(x.strip() for x in re.split(r"[,;]", missing_raw) if x.strip())

        min_cov_s = (row.get("min_coverage") or "").strip()
        min_id_s = (row.get("min_identity") or "").strip()
        sha_s = (row.get("sha256") or "").strip()

        return cls(
            source=source,
            base_dir=base_dir,
            remote_base_url=remote_base_url,
            genome=(row["genome"] or "").strip(),
            database=(row["database"] or "").strip(),
            locus=(row["locus"] or "").strip(),
            phenotype=(row["phenotype"] or "").strip(),
            typeable=cls.parse_bool(row["typeable"] or ""),
            problems=cls.parse_problems(row.get("problems") or ""),
            missing_genes=missing,
            min_coverage=float(min_cov_s) if min_cov_s else None,
            min_identity=float(min_id_s) if min_id_s else None,
            sha256=sha_s.lower() if sha_s else None,
            notes=(row.get("notes") or "").strip(),
        )

    def verify(self, result: SerotypingResult | None) -> None:
        r"""Assert that `result` satisfies all declared expectations for this row."""
        assert result is not None, f"[{self.test_id}] Serotyper returned None"
        assert result.best_locus_name == self.locus, (
            f"[{self.test_id}] Expected locus {self.locus!r}, got {result.best_locus_name!r} "
            f"(completeness={result.best_locus_completeness:.2%})"
        )
        assert result.phenotype == self.phenotype, (
            f"[{self.test_id}] Expected phenotype {self.phenotype!r}, got {result.phenotype!r}"
        )
        assert result.typeable is self.typeable, (
            f"[{self.test_id}] Expected typeable={self.typeable}, got {result.typeable} "
            f"(problems={result.problems.to_symbols().decode()!r})"
        )
        if self.problems is not None:
            assert result.problems == self.problems, (
                f"[{self.test_id}] Expected problems={self.problems!r} "
                f"({self.problems.to_symbols().decode()!r}), got {result.problems!r} "
                f"({result.problems.to_symbols().decode()!r})"
            )
        if self.missing_genes is not None:
            assert set(result.missing_expected_genes) == set(self.missing_genes), (
                f"[{self.test_id}] Expected missing_expected_genes={self.missing_genes}, "
                f"got {result.missing_expected_genes}"
            )
        if self.min_coverage is not None:
            assert result.percent_coverage >= self.min_coverage, (
                f"[{self.test_id}] Expected coverage >= {self.min_coverage}%, got {result.percent_coverage:.2f}%"
            )
        if self.min_identity is not None:
            assert result.percent_identity >= self.min_identity, (
                f"[{self.test_id}] Expected identity >= {self.min_identity}%, got {result.percent_identity:.2f}%"
            )


@dataclass(frozen=True, slots=True)
class NcbiGenomeCache:
    r"""Content-cached NCBI Datasets v2 and remote/local FASTA resolver for curated test genomes.

    Streams `GENOME_FASTA` ZIP archives from `https://api.ncbi.nlm.nih.gov/datasets/v2` via stdlib
    `urllib` + `zipfile`, caches compressed `{accession}.fna.gz` files locally under
    `~/.kaptive/genomes` (or `$KAPTIVE_CACHE_DIR/genomes`), and verifies optional SHA-256 digests.
    """

    _ACCESSION_RE: ClassVar[re.Pattern[str]] = re.compile(r"^GC[AF]_\d+\.\d+$")
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.ncbi.nlm.nih.gov/datasets/v2"

    cache_dir: Path | None = None
    timeout: float = 90.0

    @property
    def resolved_cache_dir(self) -> Path:
        r"""Target cache directory for downloaded genome FASTAs."""
        if self.cache_dir is not None:
            return self.cache_dir
        return Path(os.environ.get("KAPTIVE_CACHE_DIR", Path.home() / ".kaptive")) / "genomes"

    @classmethod
    def is_ncbi_accession(cls, spec: str) -> bool:
        r"""Return `True` if `spec` is a versioned RefSeq/GenBank assembly accession."""
        return bool(cls._ACCESSION_RE.match(spec.strip()))

    def _download_ncbi_fasta(self, accession: str) -> bytes:
        quoted = urllib.parse.quote(accession.strip())
        url = f"{self.DEFAULT_BASE_URL}/genome/accession/{quoted}/download?include_annotation_type=GENOME_FASTA"
        headers = {
            "Accept": "application/zip",
            "User-Agent": "kaptive-test-suite/3.0",
        }
        if api_key := os.environ.get("NCBI_API_KEY", "").strip():
            headers["api-key"] = api_key

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            zip_bytes = resp.read()

        if not zip_bytes.startswith(b"PK"):
            snippet = zip_bytes[:200].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"NCBI Datasets returned non-ZIP payload for {accession!r} "
                f"(verify accession version exists): {snippet}"
            )

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            fna_chunks = [
                zf.read(info.filename)
                for info in zf.infolist()
                if not info.is_dir() and info.filename.lower().endswith((".fna", ".fa", ".fasta"))
            ]
        if not fna_chunks:
            raise RuntimeError(f"No FASTA payload found inside NCBI Datasets archive for {accession!r}")
        return b"\n".join(fna_chunks)

    @staticmethod
    def _verify_sha256(raw_bytes: bytes, expected_hex: str | None, label: str) -> None:
        if expected_hex:
            digest = hashlib.sha256(raw_bytes).hexdigest().lower()
            if digest != expected_hex.lower():
                raise ValueError(f"SHA-256 mismatch for {label}: expected {expected_hex}, got {digest}")

    def resolve(self, expectation: SerotypeExpectation) -> GenomeAssembly:
        r"""Resolve `expectation.genome` from local path, `~/.kaptive/genomes` cache, NCBI, or remote repo."""
        spec = expectation.genome
        cache_root = self.resolved_cache_dir
        if self.is_ncbi_accession(spec):
            cache_root.mkdir(parents=True, exist_ok=True)
            cached_gz = cache_root / f"{spec}.fna.gz"
            if cached_gz.is_file() and cached_gz.stat().st_size > 0:
                if expectation.sha256:
                    self._verify_sha256(gzip.decompress(cached_gz.read_bytes()), expectation.sha256, spec)
                return GenomeAssembly.from_file(cached_gz)

            raw_fna = self._download_ncbi_fasta(spec)
            self._verify_sha256(raw_fna, expectation.sha256, spec)
            tmp_gz = cache_root / f".{spec}.{os.getpid()}.tmp.fna.gz"
            tmp_gz.write_bytes(gzip.compress(raw_fna, compresslevel=6))
            tmp_gz.replace(cached_gz)
            return GenomeAssembly.from_file(cached_gz)

        if expectation.base_dir is not None:
            for candidate in (expectation.base_dir / spec, expectation.base_dir / "tests" / spec):
                if candidate.is_file():
                    if expectation.sha256:
                        self._verify_sha256(candidate.read_bytes(), expectation.sha256, str(candidate))
                    return GenomeAssembly.from_file(candidate)

        if expectation.remote_base_url is not None:
            cache_root.mkdir(parents=True, exist_ok=True)
            safe_name = Path(spec).name
            cached_remote = cache_root / safe_name
            if not cached_remote.is_file():
                url = f"{expectation.remote_base_url.rstrip('/')}/{spec.lstrip('/')}"
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    payload = resp.read()
                self._verify_sha256(payload, expectation.sha256, url)
                cached_remote.write_bytes(payload)
            return GenomeAssembly.from_file(cached_remote)

        raise FileNotFoundError(f"Could not resolve genome {spec!r} declared in {expectation.source}")


@final
class CuratedTestLoader:
    r"""Discovers `test.tsv` tables and `[lolo_allowlist]` rules from decentralized database repositories."""

    REMOTE_CANDIDATES: ClassVar[tuple[str, ...]] = ("tests/test.tsv", "test.tsv")

    def __new__(cls) -> CuratedTestLoader:
        raise TypeError("CuratedTestLoader is a static namespace and cannot be instantiated.")

    @classmethod
    def parse_tsv_text(
        cls,
        text: str,
        *,
        source: str,
        base_dir: Path | None = None,
        remote_base_url: str | None = None,
    ) -> tuple[SerotypeExpectation, ...]:
        r"""Parse TSV content into an immutable tuple of `SerotypeExpectation` records."""
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if not lines:
            return ()
        reader = csv.DictReader(lines, delimiter="\t")
        return tuple(
            SerotypeExpectation.from_row(row, source=source, base_dir=base_dir, remote_base_url=remote_base_url)
            for row in reader
        )

    @classmethod
    def load_file(cls, path: Path) -> tuple[SerotypeExpectation, ...]:
        r"""Load `SerotypeExpectation` rows from a local `test.tsv` file."""
        if not path.is_file():
            return ()
        return cls.parse_tsv_text(path.read_text(encoding="utf-8"), source=str(path), base_dir=path.parent)

    @classmethod
    def load_remote_known_repos(cls, timeout: float = 15.0) -> tuple[SerotypeExpectation, ...]:
        r"""Probe unique `(owner, repo)` pairs in `DatabaseManager._KNOWN` for `tests/test.tsv` or `test.tsv`."""
        seen_repos: set[tuple[str, str]] = set()
        cases: list[SerotypeExpectation] = []
        for owner, repo, _ in DatabaseManager._KNOWN.values():
            if (owner, repo) in seen_repos:
                continue
            seen_repos.add((owner, repo))
            repo_root = f"https://raw.githubusercontent.com/{owner}/{repo}/main"
            for rel_tsv in cls.REMOTE_CANDIDATES:
                url = f"{repo_root}/{rel_tsv}"
                try:
                    with urllib.request.urlopen(url, timeout=timeout) as resp:
                        text = resp.read().decode("utf-8")
                    parent_url = url.rsplit("/", 1)[0]
                    cases.extend(cls.parse_tsv_text(text, source=url, remote_base_url=parent_url))
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                    continue
        return tuple(cases)

    @classmethod
    def discover_all(
        cls,
        *,
        tsv_files: Iterable[Path] = (),
        db_repos: Iterable[Path] = (),
        include_remote: bool = False,
    ) -> tuple[SerotypeExpectation, ...]:
        r"""Aggregate deduplicated `SerotypeExpectation` cases across `--test-tsv`, `--db-repo`, and remote repos."""
        merged: dict[tuple[str, str], SerotypeExpectation] = {}

        for tsv_path in tsv_files:
            for c in cls.load_file(tsv_path):
                merged[(c.database, c.genome)] = c

        for repo_dir in db_repos:
            for rel in cls.REMOTE_CANDIDATES:
                candidate = repo_dir / rel
                if candidate.is_file():
                    for c in cls.load_file(candidate):
                        merged[(c.database, c.genome)] = c
                    break

        if include_remote:
            for c in cls.load_remote_known_repos():
                merged[(c.database, c.genome)] = c

        return tuple(merged.values())

    @staticmethod
    def parse_allowlist_section(raw: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
        r"""Normalize a parsed TOML `[lolo_allowlist]` table or list into immutable nested tuples."""
        if raw is None:
            return ()
        if isinstance(raw, list):
            return tuple((str(item), ("*",)) for item in raw)
        if isinstance(raw, Mapping):
            rules: list[tuple[str, tuple[str, ...]]] = []
            for k, v in raw.items():
                targets = tuple(str(x) for x in v) if isinstance(v, list) else (str(v),)
                rules.append((str(k), targets))
            return tuple(rules)
        raise ValueError(f"Unsupported [lolo_allowlist] structure: {type(raw).__name__}")

    @classmethod
    def load_lolo_allowlist(
        cls,
        db_keyword: str,
        *,
        db_toml: Path | None = None,
        include_remote: bool = False,
        timeout: float = 15.0,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        r"""Load immutable `(query_pattern, allowed_patterns)` LOLO rules from `<Database>.toml`."""
        if db_toml is not None and db_toml.is_file():
            doc = tomllib.loads(db_toml.read_text(encoding="utf-8"))
            return cls.parse_allowlist_section(doc.get("lolo_allowlist"))

        if include_remote and db_keyword in DatabaseManager._KNOWN:
            owner, repo, db_base = DatabaseManager._KNOWN[db_keyword]
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{db_base}.toml"
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    doc = tomllib.loads(resp.read().decode("utf-8"))
                return cls.parse_allowlist_section(doc.get("lolo_allowlist"))
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                return ()

        return ()
