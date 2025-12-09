"""
Module for parsing and managing bacterial sequence files and data.
"""
from functools import partial
from io import BufferedIOBase, RawIOBase
from itertools import chain
from pathlib import Path
from re import compile as regex
from shutil import copyfileobj
from sys import stdin
from tempfile import NamedTemporaryFile
from typing import Union, Generator, IO, Literal, TextIO, get_args, Callable, Iterable
from warnings import warn

from . import KaptiveWarning
from .graph import Edge
from .seq import Record, Feature, Seq, NucleotideSeq, ProteinSeq, Qualifier, Location, Protein
from .utils import xopen, encode_strand

# Constants ------------------------------------------------------------------------------------------------------------
_SUPPORTED_FORMATS = Literal['fasta', 'gfa', 'genbank', 'fastq', 'gff', 'bed']
_SUPPORTED_FEATURES = frozenset({'CDS'})
_TAG2TYPE = {'f': float, 'i': int, 'Z' : str}  # See: https://github.com/GFA-spec/GFA-spec/blob/master/GFA1.md#optional-fields
_SEQUENCE_FILE_REGEX = regex(
    r'\.('
    r'(?P<fasta>f(asta|a|na|fn|as|aa))|'
    r'(?P<fastq>f(ast)?q)|'
    r'(?P<gfa>gfa)|'
    r'(?P<genbank>g(b|bff|bk|enbank))|'
    r'(?P<gff>gff(3)?)|'
    r'(?P<bed>bed)'
    r')\.?(?P<compression>(gz|bz2|xz|zst))?$'
)
# Regex for Illumina BaseSpace read files as specified here:
# https://support.illumina.com/help/BaseSpace_Sequence_Hub_OLH_009008_2/Source/Informatics/BS/NamingConvention_FASTQ-files-swBS.htm
_ILLUMINA_READ_REGEX = regex(
    r'(?P<sample_name>.+)_S(?P<sample_number>\d+)_L(?P<lane_number>\d+)_R(?P<read_number>\d+)_(?P<index_number>\d+)$')
_SHORT_READ_REGEX = regex(r'(?P<sample_name>.+)_R?(?P<read_number>[12])$')
_TOPOLOGY_REGEX = regex(r'(?i)(\bcircular\b|\bcircular\s*=\s*true\b)')
_COPY_NUMBER_REGEX = regex(r'depth=(\d+\.\d+)')
_GENBANK_LOCATION_REGEX = regex(r'(?P<partial_start><)?(?P<start>[0-9]+)\.\.(?P<partial_end>>)?(?P<end>[0-9]+)')
_GENBANK_ORIGIN_REGEX = regex(r'(?m)[AaTtCcGg]')
_CIGAR_OPERATIONS = regex(r'(?P<n>[0-9]+)(?P<operation>[MIDNSHP=X])')
_QUERY_CONSUMING_OPERATIONS = {"M", "I", "S", "=", "X"}
_TARGET_CONSUMING_OPERATIONS = {"M", "D", "N", "=", "X"}


# Classes --------------------------------------------------------------------------------------------------------------
class ParserError(Exception):
    pass


class ParserWarning(KaptiveWarning):
    pass


class SeqFileWarning(KaptiveWarning):
    pass


class SeqFileError(Exception):
    pass


class SeqFile:
    """
    Class for handling a (possibly compressed) file or stream of biological formats.

    :param file: Path to file (str/Path) or an IO stream (binary or text).
    :param format_: Format of the file. If None, the format will be guessed from the
                    filename extension or the stream content.
    :return: A SeqFile instance
    """
    def __init__(self, file: Union[str, Path, IO], format_: _SUPPORTED_FORMATS = None, temp_prefix='seqfile_'):
        self.id: str = "unknown"
        self.path: Union[Path, None] = None
        self.format: Union[str, None] = format_
        self._handle: Union[IO, None] = None
        self._from_stream: bool = False
        self._open_func: Union[None, Callable] = None

        if file in {'-', 'stdin'}:  # Handle stdin symbol, the rest of the logic should deal with the stream
            file = stdin

        if hasattr(file, 'read') and not isinstance(file, (str, Path)):  # --- Handle IO Stream Input ---
            if not isinstance(file, (BufferedIOBase, RawIOBase)):
                if buffer := getattr(file, 'buffer', None):
                    file = buffer
                else:
                    raise SeqFileError("Input stream is not binary and lacks an accessible binary buffer (.buffer).")

            # The existing, clever solution: dump the stream to a temp file to make it seekable.
            with NamedTemporaryFile(prefix=temp_prefix, suffix='.tmp', delete=False, mode='wb') as temp_f_handle:
                copyfileobj(file, temp_f_handle)
                self.path = Path(temp_f_handle.name)
                self.id = self.path.stem
                self._from_stream = True

            # Now, we can safely open and guess from the temp file.
            if self.format is None:
                try:  # xopen handles decompression automatically based on magic numbers
                    with xopen(self.path, mode='rt') as f:
                        self.format = _guess_format_from_handle(f)
                except Exception as e:  # Clean up the temp file on failure
                    self.path.unlink(missing_ok=True)
                    raise e

            # The suffix for the temp file doesn't matter since we use xopen's 'magic' method
            self._open_func = partial(xopen, self.path, method='magic', mode='rt')

        elif isinstance(file, (str, Path)):  # --- Handle File Path Input ---
            self.path = file if isinstance(file, Path) else Path(file)
            if m := _SEQUENCE_FILE_REGEX.search(self.path.name):
                self.id = self.path.name.rstrip(m.group())  # If format is not provided, guess from extension.
                self.format = format_ or next(fmt for fmt in get_args(_SUPPORTED_FORMATS) if m[fmt])
                self._open_func = partial(xopen, self.path, method=m['compression'] or 'uncompressed', mode='rt')
            else:  # If no valid extension, try guessing from content
                if self.format is None:
                    try:
                        with xopen(self.path, mode='rt') as f:
                            self.format = _guess_format_from_handle(f)
                        self._open_func = partial(xopen, self.path, method='magic', mode='rt')
                    except Exception as e:
                         raise SeqFileError(f'Unsupported file extension and could not guess format for: {self.path.name}') from e
                else: # Format was provided, but extension is weird. Trust the user.
                    self._open_func = partial(xopen, self.path, method='magic', mode='rt')
        else:
            raise TypeError(f"Input must be a file path (str or Path) or an IO stream, not {type(file)}")

    def __repr__(self):
        return f'SeqFile({self.id}, format={self.format})'

    def __str__(self):
        return self.id

    def __enter__(self):
        self._ensure_handle_open()  # Open the handle when entering context, using the appropriate _open_func
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()  # Ensure the primary handle (_handle) is closed
        if self._from_stream:  # Clean up the temporary file *only* if it was created from a stream
            try:  # Use missing_ok=True for robustness against race conditions etc.
                self.path.unlink(missing_ok=True)
            except OSError as e:  # Log or warn about failure to delete temp file
                 warn(f"Could not delete temporary file {self.path}: {e}", SeqFileWarning)

    def __del__(self):  # Minimal cleanup: try to close the handle and delete temp file if applicable
        self.close()  # Subject to usual __del__ caveats (timing, interpreter state)
        if getattr(self, '_from_stream', False) and hasattr(self, 'path'):
             try:  # Avoid print/warn in __del__ if possible, or use sys.stderr
                 self.path.unlink(missing_ok=True)
             except Exception:
                 pass # Ignore errors in __del__

    def _ensure_handle_open(self):
        """Internal helper to open/reopen the file via _open_func."""
        if self._handle is None or self._handle.closed:
             self.close()  # Close any potentially lingering closed handle first
             self._handle = self._open_func()  # Open fresh using the configured function (xopen for paths/streams)

    def close(self):
        """Closes the associated file handle (_handle), if open."""
        if self._handle and not self._handle.closed:
            try:
                self._handle.close()
            except Exception as e:
                 pass   # Ignore potential errors during close, especially in __del__ context
        self._handle = None  # Mark as closed

    def read(self) -> str:
        """Reads the entire content of the file (decompressed)."""
        self._ensure_handle_open()  # Ensure we have a fresh handle from the start
        content = self._handle.read()  # Read from the current handle (which was just opened/rewound)
        # Close handle after full read? Optional, depends on expected usage.
        # self.close()
        return content

    def rewind(self):
        """Resets the file to be read from the beginning."""
        # The simplest way to rewind when using xopen dynamically is to close
        # the current handle and let _ensure_handle_open get a new one.
        self.close()  # Next call to read() or __iter__() will automatically reopen via _ensure_handle_open()

    def __iter__(self) -> Generator[Union['Record', 'Edge', 'Feature'], None, None]:
        """Returns an iterator (generator) over records in the file."""
        self._ensure_handle_open() # Ensure handle is open/reopened
        if not self.format:
             raise SeqFileError("Cannot parse file: format is unknown.")
        yield from parse(self._handle, self.format)
        # Note: Iteration consumes the handle. Re-iteration requires rewind()/re-opening.

    def peek(self) -> str:
        """Returns the first line and resets to the beginning."""
        self._ensure_handle_open()
        first_line = self._handle.readline()
        self.rewind()  # Rewind by closing and letting the next operation reopen
        return first_line


class ReadFileError(Exception):
    pass


class ReadFile(SeqFile):
    """
    A subclass of :class:`~eris.core.io.SeqFile` that specifically handles a single file of sequencing reads
    in FASTQ format.
    This class extracts and stores read-specific information from the filename, and is designed to be used with the
    :class:`~eris.core.io.ReadSet` class.

    Attributes:
        sample_name: The name of the sample from which the reads were derived.
        read_type: The type of read ('short', 'illumina', 'long', 'pb', 'ont', or 'unknown').
        sample_number: The sample number (for Illumina reads).
        lane_number: The lane number (for Illumina reads).
        read_number: The read number (1 or 2 for paired-end reads).
        index_number: The index number (for Illumina reads).
    """
    def __init__(self, file: Union[str, Path]):
        """
        :param file: Path to file (str/Path) or an IO stream (binary or text).
        """
        super().__init__(file)
        self.sample_name = self.id
        self._read_type: Literal['unknown', 'short', 'illumina', 'long', 'pb', 'ont'] = 'unknown'
        self._sample_number: int = 0
        self._lane_number: int = 0
        self._read_number: int = 0
        self._index_number: int = 0
        if m := _ILLUMINA_READ_REGEX.match(self.id):
            self.sample_name = m.group('sample_name')
            self._read_type = "illumina"
            self._sample_number = int(m.group('sample_number'))
            self._lane_number = int(m.group('lane_number'))
            self._read_number = int(m.group('read_number'))
            self._index_number = int(m.group('index_number'))
        elif m := _SHORT_READ_REGEX.match(self.id):
            self.sample_name = m.group('sample_name')
            self._read_type = "short"
            self._read_number = int(m.group('read_number'))

    @property
    def read_type(self):
        return self._read_type

    @property
    def sample_number(self):
        return self._sample_number

    @property
    def lane_number(self):
        return self._lane_number

    @property
    def read_number(self):
        return self._read_number

    @property
    def index_number(self):
        return self._index_number


class ReadSetError(Exception):
    pass


class ReadSet:
    """
    Whilst still representing genomes, these are different from Genome instances as they will not hold
    sequence information in memory and may consist of multiple files.
    """
    def __init__(self, *files: Union[str, Path, ReadFile], id_: str = None, reference: 'Genome' = None):
        self.id = id_
        self._files = []
        self.reference = reference
        read_types = set()
        for file in files:
            if not isinstance(file, ReadFile):
                file = ReadFile(file)
            if self.id is None:
                self.id = file.sample_name
            else:
                if self.id != file.sample_name:
                    raise ReadSetError(f'{file.id} sample name {file.sample_name} does not match ReadSet {self.id}')
            self._files.append(file)
            read_types.add(file.read_type)
        if len(read_types) > 1:
            raise ReadSetError(f'Hybrid read set {self.id} not yet supported')
        else:
            self._set_type = read_types.pop()

    @property
    def set_type(self):
        return self._set_type

    def __iter__(self):
        return iter(self._files)

    def __repr__(self):
        return self.id

    def __str__(self):
        return ' '.join([str(read) for read in self._files])

    def __len__(self):
        return len(self._files)


# Functions ------------------------------------------------------------------------------------------------------------
def _guess_format_from_handle(handle: TextIO) -> _SUPPORTED_FORMATS:
    """
    Guesses the file format by peeking at the first line of a text handle.
    Assumes the handle is seekable.
    """
    try:
        first_line = handle.readline().strip()
        handle.seek(0)  # Rewind the handle for the actual parser

        if not first_line:
            raise SeqFileError("Cannot guess format from an empty file.")

        if first_line.startswith('>'):
            return 'fasta'
        elif first_line.startswith('@'):
            return 'fastq'
        elif first_line.startswith('LOCUS'):
            return 'genbank'
        elif first_line.startswith('S\t'):
            return 'gfa'
        elif first_line.startswith('##gff-version 3'):
            return 'gff'
        else:
            raise SeqFileError(f"Could not guess file format from first line: '{first_line[:100]}...'")

    except Exception as e:
        raise SeqFileError("Failed to read from stream to guess format.") from e


def parse(handle: TextIO, format_: _SUPPORTED_FORMATS = 'guess', *args, **kwargs) -> Generator[
    Union[Record, Edge, Feature], None, None]:
    """
    Simple parser for fasta, gfa and genbank formats,
    similar to Biopython <https://biopython.org/docs/latest/api/Bio.SeqIO.html#Bio.SeqIO.parse>

    Parameters:
        handle: A file handle opened in text-mode / text stream
        format_: The format of the file; must be one of the supported formats or guess by default

    Yields:
        Record, Feature or Edge objects
    """
    if format_ == 'guess':
        if m := _SEQUENCE_FILE_REGEX.search(handle.name):
            format_ = next(fmt for fmt in get_args(_SUPPORTED_FORMATS) if m[fmt])
        else:
            raise SeqFileError(f'Unsupported SeqFile format or extension: {handle.name}')
    if parser := {'fasta': _parse_fasta, 'gfa': _parse_gfa, 'genbank': _parse_genbank}.get(format_):
        yield from parser(handle, *args, **kwargs)
    else:
        raise NotImplementedError(f'Format "{format_}" not supported')


def _parse_fasta(handle: TextIO) -> Generator[Record, None, None]:
    """
    Simple FASTA parser

    :param handle: A file handle opened in text-mode / text stream
    :returns: A Generator of Record objects
    """
    header, line_buffer, record = '', [], None
    for line in chain(handle, ['>']):
        if not (line := line.strip()):
            continue
        if line.startswith('>'):
            if header and line_buffer:
                name, _, desc = header.partition(' ')

                _COPY_NUMBER_REGEX.match(desc)

                yield (record := Record(Seq(''.join(line_buffer)), name, desc, qualifiers=[
                    Qualifier('topology', 'circular' if _TOPOLOGY_REGEX.search(desc) else 'linear'),
                    Qualifier('depth', float(match[1]) if (match := _COPY_NUMBER_REGEX.search(desc)) else 1)
                ]))
            header, line_buffer = line[1:], []
        else:
            line_buffer.append(line)
    if record is None:
        warn('No records parsed', ParserWarning)


def _parse_fastq(handle: TextIO) -> Generator[Record, None, None]:
    """
    Simple FASTQ parser

    :param handle: A file handle opened in text-mode / text stream
    :returns: A Generator of Record objects
    """
    record = None
    while True:
        if not (header := handle.readline().strip()):  # 1. Read the header line
            break  # End of file
        if not header.startswith('@'):
            raise ValueError(f"Expected FASTQ record header starting with '@', but got: {header}")
        if not (seq := handle.readline().strip()):  # 2. Read the sequence line
            raise ValueError(f"Unexpected end of file after reading header: {header}")
        if not (sep := handle.readline().strip()):  # 3. Read the separator line
            raise ValueError(f"Unexpected end of file after reading sequence for header: {header}")
        if not sep.startswith('+'):
            raise ValueError(f"Expected FASTQ separator line starting with '+', but got: {sep}")
        if not (qual := handle.readline().strip()):  # 4. Read the quality line
            raise ValueError(f"Unexpected end of file after reading separator for header: {header}")
        name, desc = header.split(' ', 1) if ' ' in header else (header, '')
        yield (record := Record(NucleotideSeq(seq), name[1:], desc, qualifiers=[Qualifier('quality', qual)]))
    if record is None:
        warn('No records parsed', ParserWarning)


def _parse_gfa(handle: TextIO) -> Generator[Union[Record, Edge], None, None]:
    """
    Simple GFA parser, see https://gfa-spec.github.io/GFA-spec/GFA1.html

    :param handle: A file handle opened in text-mode / text stream
    :returns: A Generator of Record or Edge objects
    """
    record = None
    for line in handle:  # Iterate over file lines
        if line.startswith('S\t'):  # Segment contains contig info
            name, seq, tags = line[2:].strip().split('\t', 2)  # Split into 3 parts: name, sequence and description
            if len(seq) >= 1:  # Check sequence is at least 1bp
                yield (record := Record(seq=NucleotideSeq(seq), id_=name, qualifiers=list(parse_tags(tags.split('\t')))))
        elif line.startswith('L\t'):  # Add links once all contigs are added
            u, from_strand, v, to_strand, cigar = line[2:].strip().split('\t')[:5]
            yield Edge(u, v, {'u strand': encode_strand(from_strand), 'v strand': encode_strand(to_strand),
                              'overlap': next((n for op, n, _, _, _ in parse_cigar(cigar) if op == 'M'), 0)})
    if record is None:
        warn('No records parsed', ParserWarning)


def _parse_bed(handle: TextIO, feature_kinds: set[str] = _SUPPORTED_FEATURES) -> Generator[Feature, None, None]:
    """
    Simple BED parser, see https://samtools.github.io/hts-specs/BEDv1.pdf

    :param handle: A file handle opened in text-mode / text stream
    :param feature_kinds: Set of feature kinds to parse
    :returns: A Generator of Feature objects
    """
    feature = None
    for line in handle:  # Note, this only supports BED10
        if len(parts := line.strip().split('\t')) >= 10 and parts[7] in feature_kinds:
            feature = Feature(
                Location(int(parts[1]), int(parts[2]), -1 if parts[5] == '-' else 1, parent_id=parts[0]),
                kind=parts[7],
                qualifiers=[Qualifier(*i.split('=', 1)) for i in parts[9].split(';')] + [
                    Qualifier('source', parts[6])]
            )
            feature.id = feature['ID'] or 'unknown'  # Update the feature.id using the ID qualifier
            yield feature

    if feature is None:
        warn('No features parsed', ParserWarning)


def _parse_gff(handle: TextIO, feature_kinds: set[str] = _SUPPORTED_FEATURES) -> Generator[Feature, None, None]:
    """
    Simple GFF3 parser, see https://ensembl.org/info/website/upload/gff3.html

    :param handle: A file handle opened in text-mode / text stream
    :param feature_kinds: Set of feature kinds to parse
    :returns: A Generator of Feature objects
    """
    feature = None
    for line in handle:
        if line.startswith('#'):
            continue
        if len(parts := line.strip().split('\t')) >= 9 and parts[2] in feature_kinds:
            feature = Feature(
                Location(int(parts[3]) - 1, int(parts[4]), -1 if parts[6] == '-' else 1, parent_id=parts[0]),
                kind=parts[2],
                qualifiers=[Qualifier(*i.split('=', 1)) for i in parts[8].split(';')] + [
                    Qualifier('source', parts[1])]
            )
            feature.id = feature['ID'] or 'unknown'  # Update the feature.id using the ID qualifier
            yield feature
    if feature is None:
        warn('No features parsed', ParserWarning)


def _parse_genbank(handle: TextIO, feature_kinds: set[str] = _SUPPORTED_FEATURES) -> Generator[Record, None, None]:
    """
    Simple genbank parser

    :param handle: A file handle opened in text-mode / text stream
    :param feature_kinds: Set of feature kinds to parse; note the 'source' feature will populate the record's qualifiers
    :returns: A Generator of Record objects
    """
    line_buffer, record = [], None
    for line in handle:  # Loop over lines in chunk
        if line := line.strip():
            if line.startswith('LOCUS'):  # This is the beginning of the new record
                if len(line_buffer) > 1:  # Records must all consist of more than 1 line
                    yield (record := _parse_genbank_record(line_buffer, feature_kinds))  # Yield the previous record
                    line_buffer = []
            if not line.startswith('//'):  # Add lines until the end of the record, signified by "//"
                line_buffer.append(line)
    if len(line_buffer) > 1:  # Records must all consist of more than 1 line
        yield _parse_genbank_record(line_buffer, feature_kinds)
    if record is None:
        warn('No records parsed', ParserWarning)


def _parse_genbank_record(line_buffer: list[str], feature_kinds: set[str] = _SUPPORTED_FEATURES) -> Record:
    """Parser for a single record in Genbank format"""
    features, origin = '\n'.join(line_buffer).split('FEATURES', 1)[1].split('\nORIGIN\n', 1)
    record = Record(
        # NucleotideSeq(''.join(chain.from_iterable(i.split()[1:] for i in origin if i))),
        # The original code above returned empty lists for some reason so using regex for now; unsure if faster
        NucleotideSeq(''.join(_GENBANK_ORIGIN_REGEX.findall(origin))),
        line_buffer[0].split()[1], line_buffer[1].split(maxsplit=1)[1],
        qualifiers=[Qualifier('topology', 'circular' if _TOPOLOGY_REGEX.search(line_buffer[0]) else 'linear')]
    )
    current_feature = []
    for line in features.split('\n')[1:]:
        if not line.startswith('/') and len(line.split()) == 2 and '..' in line:  # New feature
            if current_feature:
                if feature := _parse_genbank_feature(current_feature, feature_kinds, record.id):
                    if feature.kind == 'source':
                        record.qualifiers.extend(feature.qualifiers)
                    else:
                        record.features.append(feature)
                current_feature = []
        current_feature.append(line)
    if current_feature:
        if feature := _parse_genbank_feature(current_feature, feature_kinds, record.id):
            if feature.kind == 'source':
                record.qualifiers.extend(feature.qualifiers)
            else:
                record.features.append(feature)
    return record


def _parse_genbank_feature(line_buffer: list[str], feature_kinds: set[str] = frozenset({'CDS'}), parent_id: str = None
                           ) -> Union[Feature, None]:
    feature_kind, location = line_buffer[0].split(maxsplit=1)
    if feature_kind not in feature_kinds and feature_kind != 'source':
        return None
    feature = Feature(kind=feature_kind, location=_parse_genbank_location(location, parent_id))
    if len(line_buffer) > 1:
        for qualifier in _parse_genbank_qualifiers(line_buffer[1:]):
            if qualifier.key == 'locus_tag' and feature.id == 'unknown':  # Use locus tag as ID
                feature.id = qualifier.value  # Don't add as qualifier
            elif qualifier.key == 'translation':  # Turn the translation into a Seq object
                if not qualifier.value.endswith(Protein.stop_symbol):
                    qualifier.value += Protein.stop_symbol  # Add a stop symbol if missing
                qualifier.value = ProteinSeq(qualifier.value)
                feature.qualifiers.append(qualifier)
            else:
                feature.qualifiers.append(qualifier)
    return feature


def _parse_genbank_location(location: str, parent_id: str = None) -> Location:
    locations = []
    strand: Literal[1, -1] = -1 if 'complement' in location else 1
    for match in _GENBANK_LOCATION_REGEX.finditer(location):
        locations.append(
            new_location := Location(int(match.group('start')) - 1, int(match.group('end')), strand,
                                     parent_id=parent_id))
        if match.group('partial_start'):
            new_location.partial_start = True
        if match.group('partial_end'):
            new_location.partial_end = True
    if not locations:
        raise ValueError(f'Could not parse location: {location}')

    location = locations.pop(0)  # type: Location
    return location


def _parse_genbank_qualifiers(lines: list[str]) -> Generator[Qualifier, None, None]:
    """Parse the attribute lines of a genbank record"""
    current_qualifier = []
    for line in lines:
        if line.startswith('/'):
            if current_qualifier:
                yield _format_qualifier(current_qualifier)
            current_qualifier = []
        current_qualifier.append(line)
    if current_qualifier:
        yield _format_qualifier(current_qualifier)


def _format_qualifier(qualifier: list[str]) -> Qualifier:
    # So far, I think only multiline translations need to be joined without whitespace
    # TODO: We could add a step to replace multiple spaces with a single space
    qualifier = ('' if qualifier[0].startswith('/translation') else ' ').join(qualifier).lstrip('/')
    if '=' not in qualifier:
        return Qualifier(qualifier, True)  # I prefer having "True" as the value rather than None
    key, value = qualifier.split('=', 1)
    if value.startswith('"'):
        value = value.strip('"')
    elif value.startswith('('):
        value = value.lstrip('(').rstrip(')')
    elif '.' in value:
        value = float(value)
    else:
        value = int(value)
    return Qualifier(key, value)


def parse_tags(cols: Iterable[str], tag_delimiter: str = ':') -> Generator[Qualifier, None, None]:
    """Parse tag column and yield a tuple of the tag and value in the correct type"""
    for item in cols:
        tag, typ, val = item.split(tag_delimiter, maxsplit=2)  # type: str, str, str
        if (tag_lower := tag.lower()) == 'dp':
            tag = 'depth'
        elif tag_lower == 'ln':
            tag = 'length'
        elif tag_lower == 'kc':
            tag = 'kmer count'
        yield Qualifier(tag, _TAG2TYPE.get(typ, str)(val))


def parse_cigar(cigar: str) -> Generator[tuple[Literal['M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X'], int, int, int, int], None, None]:
    """
    Parses the cigar with a regular expression

    Parameters:
        The cigar string to parse

    Yields:
        A tuple of (operation, length, query_length, target_length, alignment_length)
        for each operation in the cigar string
    """
    if cigar:
        query_len, target_len, ali_len = 0, 0, 0
        for match in _CIGAR_OPERATIONS.finditer(cigar):
            if not match:
                raise ValueError(f'Could not parse cigar: {cigar}')
            op, n = match['operation'], int(match['n'])
            ali_progressed = False
            if op in _QUERY_CONSUMING_OPERATIONS:
                query_len += n
                ali_progressed = True
            if op in _TARGET_CONSUMING_OPERATIONS:
                target_len += n
                ali_progressed = True
            if ali_progressed:
                ali_len += n
            yield op, n, query_len, target_len, ali_len
