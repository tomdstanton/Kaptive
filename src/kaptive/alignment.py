"""
Module for parsing, viewing and managing sequence alignments.
"""
from typing import Iterable, Generator, Union, Any, Callable
from operator import attrgetter

from .seq import HasLocation, Location, Seq, Qualifier, Feature
from .utils import encode_strand
from .io import parse_tags

# Classes --------------------------------------------------------------------------------------------------------------
class AlignmentError(Exception):
    pass


class Alignment(HasLocation):
    """
    Class representing an alignment between a query sequence and target sequence.

    Attributes:
        query: Query sequence name
        query_length: Query sequence length
        query_start: Query start coordinate (0-based)
        query_end: Query end coordinate (0-based)
        target: Target sequence name
        target_length: Target sequence length
        length: Number of residues in the alignment including gaps
        cigar: CIGAR string
        score: Alignment score
        E: Alignment entropy
        n_matches: Number of matching residues in the alignment
        quality: Mapping quality (0-255 with 255 for missing)
        qualifiers: {tag: value} pairs
        identity: Percentage of matching residues
        query_coverage: Percentage of query sequence covered by the alignment
        target_coverage: Percentage of target sequence covered by the alignment
        aligned_seqs: Tuple of aligned sequences
    """

    def __init__(self, query: str, query_start: int, query_end: int, location: Location,
                 query_length: int = 0, target_length: int = 0, length: int = 0, cigar: str = None,
                 n_matches: int = 0, quality: int = 0, qualifiers: list[Qualifier] = None, score: float = 0,
                 E: float = 0, identity: float = 0,  query_coverage: float = 0, target_coverage: float = 0,
                 aligned_seqs: tuple[Seq, Seq] = None):
        super().__init__(location)
        self.query: str = query  # Query sequence name
        self.query_length: int = query_length  # Query sequence length
        self.query_start: int = query_start  # Query start coordinate (0-based)
        self.query_end: int = query_end  # Query end coordinate (0-based)
        self.target_length: int = target_length  # Target sequence length
        self.length: int = length  # Number of residues in the alignment including gaps
        self.cigar: str = cigar
        self.score: float = score
        self.E: float = E
        self.n_matches: int = n_matches  # Number of matching residues in the alignment
        self.quality: int = quality  # Mapping quality (0-255 with 255 for missing)
        self.qualifiers: list[Qualifier] = qualifiers or []
        self.identity: float = identity
        self.query_coverage: float = query_coverage
        self.target_coverage: float = target_coverage
        self.aligned_seqs: tuple[Seq, Seq] = aligned_seqs

    def __repr__(self):
        return f'{self.query}_{self.query_start}_{self.query_end}:{self.location}'

    def __len__(self):
        return self.length

    def __getitem__(self, item: str) -> Union[Any, None]:
        """
        Quick method of getting a qualifier by key using next()
        Warning:
            This will only return the first instance, which in most cases is fine
        """
        if not isinstance(item, str):
            raise TypeError(item)
        return next((v for k, v in self.qualifiers if k == item), None)

    def flip(self) -> 'Alignment':
        """Flips the query and the target"""
        return Alignment(
            query=self.location.parent_id,
            query_start=self.location.start,
            query_end=self.location.end,
            query_length=self.target_length,
            location=Location(self.query_start, self.query_end, self.location.strand, self.location.partial_start,
                              self.location.partial_end, self.query),
            target_length=self.query_length,
            length=self.length,
            cigar=self.cigar,
            score=self.score,
            E=self.E,
            n_matches=self.n_matches,
            quality=self.quality,
            qualifiers=self.qualifiers,
            identity=self.identity,
            query_coverage=self.target_coverage,
            target_coverage=self.query_coverage,
            aligned_seqs=None  # self.aligned_seqs
        )

    def as_feature(self, feature_kind: str, extra_qualifiers: list[str] = None) -> Feature:
        feature = Feature(self.location, feature_kind, self.qualifiers)
        if extra_qualifiers:
            feature.qualifiers.extend(Qualifier(*i) for i in zip(extra_qualifiers, attrgetter(*extra_qualifiers)(self)))
        return feature

    @classmethod
    def from_paf(cls, line: str):
        """
        Parse a line in PAF format and return an Alignment object.

        Parameters:
            line: A text string representing a single alignment
        """
        if len(line := line.strip().split('\t')) < 12:
            raise AlignmentError(f"PAF Line has < 12 columns: {line}")
        try:
            self = Alignment(  # Parse standard fields
                query=line[0], query_length=int(line[1]), query_start=int(line[2]), query_end=int(line[3]),
                location=Location(int(line[7]), int(line[8]), encode_strand(line[4]), parent_id=line[5]),
                target_length=int(line[6]), n_matches=int(line[9]), length=int(line[10]),
                quality=int(line[11]), qualifiers=list(parse_tags(line[12:]))
            )
            self.identity = self.n_matches / self.length
            self.query_coverage = self.length / self.query_length
            self.target_coverage = self.length / self.target_length
            self.cigar = self['cg']
            self.score = self['AS'] or 0
            is_partial_at_query_start = self.query_start > 0
            is_partial_at_query_end = self.query_end < self.query_length
            if self.location.strand == 1:
                self.location.partial_start = is_partial_at_query_start
                self.location.partial_end = is_partial_at_query_end
            else:
                self.location.partial_start = is_partial_at_query_end
                self.location.partial_end = is_partial_at_query_start
            return self
        except Exception as e:
            raise AlignmentError(f"Error parsing PAF line: {line}: {e}")


# Functions ------------------------------------------------------------------------------------------------------------
def cull(keep: Alignment, alignments: Iterable[Alignment], max_overlap_fraction: float = 0.1
         ) -> Generator[Alignment, None, None]:
    """
    Filters alignments by excluding those that overlap a reference alignment beyond a specified fraction.

    This function takes a reference alignment and iterates through a collection of
    alignments, yielding only those alignments whose target does not match the
    target of the reference alignment, or whose overlap ratio with the reference
    alignment (calculated as the overlap divided by the length of the alignment)
    is less than a provided maximum overlap fraction.

    Args:
        keep (Alignment): The reference alignment used for comparison.
        alignments (Iterable[Alignment]): A collection of alignments to be filtered.
        max_overlap_fraction (float): The maximum allowed fraction of overlap
            between the reference alignment and any given alignment. Defaults to 0.1.

    Yields:
        Generator[Alignment, None, None]: A generator that yields alignments
            meeting the specified criteria.
    """
    for alignment in alignments:
        if (alignment.location.parent_id != keep.location.parent_id or
                (alignment.overlap(keep) / alignment.length) < max_overlap_fraction):
            yield alignment


def cull_all(alignments: Iterable[Alignment], key='n_matches', reverse_sort: bool = True) -> list[Alignment]:
    """
    Sort and filter a collection of alignments based on a specified key. The function sorts
    alignments by the given key and iteratively compares the alignments, keeping those that
    meet certain criteria and removes others.

    Arguments:
        alignments (Iterable[Alignment]): A collection of Alignment objects to be processed.
        key (str): The attribute name used for sorting alignments. Defaults to 'n_matches'.
        reverse_sort (bool): If True, sort alignments in descending order using the specified
            key. Defaults to True.

    Returns:
        list[Alignment]: A filtered and sorted list of alignments, meeting specified criteria.
    """
    kept_alignments = []
    sorted_alignments = sorted(alignments, key=attrgetter(key), reverse=reverse_sort)
    while sorted_alignments:
        kept_alignments.append(sorted_alignments.pop(0))
        sorted_alignments = list(cull(kept_alignments[-1], sorted_alignments))
    return kept_alignments


def cull_filtered(predicate: Callable, alignments: Iterable[Alignment]) -> Generator[Alignment, None, None]:
    """
    Cull and flatten alignments that don't overlap with alignments matching the predicate.
    """
    kept_alignments, other = [], []
    for alignment in alignments:
        (kept_alignments if predicate(alignment) else other).append(alignment)
    other = cull_all(other)  # Remove conflicting other alignments
    for alignment in kept_alignments:  # Remove other alignments overlapping best match gene alignments
        other = list(cull(alignment, other))
        yield alignment
    yield from other
