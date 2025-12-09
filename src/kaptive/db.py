"""
Copyright 2023 Tom Stanton (tomdstanton@gmail.com)
https://github.com/klebgenomics/Kaptive

This file is part of Kaptive. Kaptive is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. Kaptive is distributed
in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with Kaptive.
If not, see <https://www.gnu.org/licenses/>.
"""
from json import loads as json_loads
from pathlib import Path
from re import compile as regex
from typing import Generator, TextIO, Union, Pattern, Literal, Iterable

from kaptive import RESOURCES
from src.kaptive.io import parse
from src.kaptive.seq import Record, Feature
from src.kaptive.utils import grouper

# Constants ------------------------------------------------------------------------------------------------------------
_LOCUS_REGEX = regex(r'(?<=locus:)\w+|(?<=locus: ).*')
_TYPE_REGEX = regex(r'(?<=type:)\w+|(?<=type: ).*')
_LOGIC_PRIMARY_DELIM, _LOGIC_SECONDARY_DELIM, _LOGIC_TERTIARY_DELIM = '\t', ';', '='
_SUPPORTED_STATES = {'extra', 'truncated'}
_METADATA_FIELDS = {'version', 'keywords', 'id_threshold', 'doi', 'contact', 'repository'}
_LOGIC_HEADER = 'loci\tgenes\tphenotype\n'
_UNITS_HEADER = 'gene\tunit\n'


# Classes --------------------------------------------------------------------------------------------------------------
class InstalledDatabases:
    def __init__(self):
        self.path: Path = RESOURCES.data
        self.databases = {}
        self.metadata = {}
        self.keywords = {}
        for database, files in grouper(self.path.iterdir(), 'stem'):
            self.databases[database] = (files := {suffix[1:]: next(f) for suffix, f in grouper(files, 'suffix')})
            if 'gbk' not in files:
                raise FileNotFoundError(f'Could not find genbank file for database {database}')
            if not (metadata := files.get('json')):
                raise FileNotFoundError(f'No metadata file found for {database}')
            self.metadata[database] = (metadata := _parse_metadata(metadata))
            for keyword in metadata['keywords']:
                self.keywords[keyword] = database

    def __len__(self):
        return len(self.databases)


class Phenotype:
    def __init__(self, id_: str, loci: Pattern, states: dict[str, Pattern], priority: int = 0):
        """
        This class represents a single phenotype and the gene states that encode it.

        Each state must be satisfied (pattern returns True)
        """
        self.id: str = id_
        self.loci: Pattern = loci
        self.states: dict[str, Pattern] = states
        self.priority = priority

    def __repr__(self):
        return f'Phenotype({self.id}; loci={self.loci.pattern})'

    def __str__(self):
        return self.id


class LocusRepresentation:
    """
    Represents the gene architecture of a locus as a DAG (gene_order) and a set (gene_set)
    """

    def __init__(self, id_: str, gene_order: tuple[str] = None, gene_set: frozenset[str] = None):
        self.id = id_
        self.gene_order: tuple[str] = gene_order or ()  # Essentially a DAG of the locus
        self.gene_set: frozenset[str] = gene_set or frozenset(
            self.gene_order)  # Holds set of gene names for non-positional comparison

    def __hash__(self):
        return hash(self.gene_set)

    def __len__(self):
        return len(self.gene_order)

    def __repr__(self):
        return f"{self.id}: {self.gene_order}"

    def __eq__(self, item: Union['LocusRepresentation', tuple[str]]) -> bool:
        if isinstance(item, LocusRepresentation):
            return self.gene_set == item.gene_set
        elif isinstance(item, Iterable):
            return self.gene_set == frozenset(item)
        else:
            raise TypeError(item)

    def is_order_variant(self, gene_order: tuple[str]) -> bool:
        return self.gene_set == frozenset(gene_order) and self.gene_order != gene_order

    def is_strand_variant(self, gene_order: tuple[str]) -> bool:
        return self.gene_set == frozenset(gene_order) and self.gene_order == gene_order[::-1]


class DatabaseError(Exception):
    pass


class Database:
    """
    Databases can be instantiated in various ways. You can create an empty instance for creating a new database;
    supply an iterable of locus Record objects; and/or supply a keyword or Path to a database file.
    """

    def __init__(
            self, id_: str, loci: dict[str, Record] = None, genes: dict[str, Feature] = None,
            extra_genes: dict[str, Feature] = None, phenotypes: dict[dict[str]] = None,
            metadata: dict[str, str] = None, antigenic_units: dict[str, str] = None,
            expected_units: dict[str, str] = None, representations: dict[str, LocusRepresentation] = None
    ):
        """Represents a Kaptive database to be loaded into memory from a file or created from scratch"""
        self.id = id_
        self.loci = loci or {}
        self.genes = genes or {}
        self.extra_genes = extra_genes or {}
        # self.clusters = {}
        self.phenotypes = phenotypes or {}
        self.metadata = metadata or {}
        self.antigenic_units = antigenic_units or {}
        self.expected_units = expected_units or {}
        self.representations = representations or {}

    @classmethod
    def from_package(cls, name: str):
        """Load an installed database using its name or keyword"""
        if len(dbs := InstalledDatabases()) == 0:
            raise DatabaseError('No databases installed')
        if not (files := dbs.databases.get(name := dbs.keywords.get(name, name))):
            raise DatabaseError(f'Invalid database name or keyword: {name}')
        self = cls(name, metadata=dbs.metadata[name])
        self._load_files(files['gbk'], files.get('logic'), files.get('units'))
        return self

    @classmethod
    def from_file(cls, file: Union[str, Path]):
        """Load an external database using its path"""
        if not isinstance(file, Path):
            file = Path(file)
        if not file.is_file() or file.stat().st_size == 0:
            raise FileNotFoundError(f'File {file} not found or empty')
        name = file.stem
        if not (metadata := (files := {i.suffix[1:]: i for i in file.parent.glob(f'{name}*')}).get('json')):
            raise DatabaseError('No metadata file found')
        self = cls(name, metadata=_parse_metadata(metadata))
        self._load_files(files['gbk'], files.get('logic'), files.get('units'))
        return self

    def _load_files(self, genbank: Path, logic: Path = None, units: Path = None):
        with open(genbank, 'rt') as handle:
            for locus in _parse_database(handle):  # type: Record
                if not (extra := locus.id.startswith('Extra_genes')):
                    gene_dict = self.genes
                    if locus.id in self.loci:
                        raise DatabaseError(f'{locus} already exists in {self.id}')
                    self.loci[locus.id] = locus
                else:
                    gene_dict = self.extra_genes

                gene_names = []
                for locus_gene in locus.features:  # type: Feature
                    if locus_gene.id in gene_dict:
                        raise DatabaseError(f'{locus_gene} already exists in {self.id}')

                    gene_dict[locus_gene.id] = locus_gene  # Add locus gene to database dictionary
                    locus_gene.extract(parent=locus, store_seq=True)
                    locus_gene.translate(parent=locus, store_translation=True)

                    if not extra:
                        gene_names.append(locus_gene['gene'])

                if gene_names:
                    self.representations[locus.id] = LocusRepresentation(locus.id, tuple(gene_names), frozenset(gene_names))

        if logic:
            with open(logic, 'rt') as handle:
                for phenotype in _parse_logic(handle):
                    for locus in self.loci.values():
                        if phenotype.loci.match(locus.id):
                            states = {}
                            for s, p in phenotype.states.items():
                                states[s] = {i.id for i in (locus.features if s == 'truncated' else
                                                            self.extra_genes.values()) if p.match(i.id)}
                            self.phenotypes.setdefault(locus.id, {})[phenotype.id] = states

        if units:
            with open(units, 'rt') as handle:
                u = dict(_parse_units(handle))
                for locus in self.loci.values():
                    expected_units = set()
                    for gene in locus.features:
                        if unit := u.get(gene['gene']):
                            self.antigenic_units[gene.id] = unit
                            expected_units.add(unit)
                    self.expected_units[locus.id] = expected_units

    def __repr__(self):
        return self.id

    def __len__(self) -> int:
        return len(self.loci)

    def __iter__(self):
        return iter(self.loci.values())

    def __format__(self, __format_spec: Literal['fasta', 'fna', 'ffn', 'faa', 'bed'] = ''):
        if __format_spec == '':
            return self.__str__()
        elif __format_spec in {'fasta', 'fna', 'ffn', 'faa', 'bed'}:
            return ''.join(format(i, __format_spec) for i in self.loci.values())
        raise NotImplementedError(f'Invalid format: {__format_spec}')



# Functions ------------------------------------------------------------------------------------------------------------
def _parse_database(handle: TextIO, locus_regex: Pattern = _LOCUS_REGEX, type_regex: Pattern = _TYPE_REGEX,
                    locus_filter: Pattern = None) -> Generator[Record, None, None]:
    """
    Parses a Kaptive database genbank file; this is mostly for extracting correct locus/gene nomenclature
    from genbank records and will become redundant moving towards a stricter system.

    :param handle: A file handle opened in text-mode / text stream.
    :param locus_filter: A regular expression to filter loci by.
    :returns: A Generator of locus ``Record`` objects.
    """
    for locus in parse(handle, 'genbank'):
        locus_name, type_name = set(), set()
        if not (notes := [v for k, v in locus.qualifiers if k == 'note']):
            raise DatabaseError(f"{locus=} has no 'note' qualifiers")

        for note in notes:  # Get locus and type names from note attributes using the regexes
            if match := type_regex.search(note):
                type_name.add(match.group())
            if note.startswith('Extra genes'):  # "Extra genes: gmlABD" -> "Extra_genes_gmlABD"
                locus_name.add(f"Extra_genes_{note.split(' ')[-1]}")
            elif match := locus_regex.search(note):
                locus_name.add(match.group())

        if len(locus_name) > 1:  # Validate locus name
            raise DatabaseError(f'Found multiple locus names in {locus=}: {", ".join(locus_name)}')
        elif len(locus_name) == 0:
            raise DatabaseError(f'Could not parse locus name from {locus=}')

        locus.id = locus_name.pop()  # Update locus id
        if locus_filter and not locus_filter.search(locus.id):
            continue  # Skip this locus
        locus.desc = type_name.pop() if type_name else f'unknown ({locus.id})'  # Replace desc with type for easy access

        for n, gene in enumerate(locus.features, start=1):  # Iterate over genes and rename them
            # Note: If databases have been created correctly, this step isn't necessary
            gene.id = f"{locus.id}_{n:02d}" + (f"_{gene_name}" if (gene_name := gene['gene']) else '')
            gene.location.parent_id = locus.id  # Update gene ref

        yield locus


def _parse_logic(handle: TextIO) -> Generator[Phenotype, None, None]:
    """
    Parses a Kaptive database logic file

    :param handle: A file handle opened in text-mode / text stream.
    :returns: A Generator of locus ``Phenotype`` objects.
    """
    if (header := next(handle)) != _LOGIC_HEADER:
        raise DatabaseError(f'Phenotype logic file {handle.name} has invalid header: {header}')

    for line in handle:
        loci, states, phenotype = line.strip().split(_LOGIC_PRIMARY_DELIM)
        states = {
            (x := i.split(_LOGIC_TERTIARY_DELIM, 1))[0]: regex(x[1]) for i in states.split(_LOGIC_SECONDARY_DELIM)
        }
        for state in states:
            if state not in _SUPPORTED_STATES:
                raise DatabaseError(f'{state=} not supported; supported states: {_SUPPORTED_STATES}')
        yield Phenotype(phenotype, regex(loci), states)


def _parse_units(handle: TextIO) -> Generator[tuple[str, str], None, None]:
    """
    Parses a Kaptive database logic file

    :param handle: A file handle opened in text-mode / text stream.
    :returns: A Generator of locus ``Phenotype`` objects.
    """
    if (header := next(handle)) != _UNITS_HEADER:
        raise DatabaseError(f'Antigenic units file {handle.name} has invalid header: {header}')

    for line in handle:
        gene, unit = line.strip().split(_LOGIC_PRIMARY_DELIM)
        yield gene, unit


def _parse_metadata(path: Path) -> dict:
    """
    Parses a Kaptive database metadata JSON file

    :param path: The path to the metadata JSON file.
    :returns: A dictionary containing the metadata.
    """
    metadata = json_loads(path.read_text())
    if unexpected_fields := set(metadata.keys()).difference(_METADATA_FIELDS):
        raise DatabaseError(
            f'Unexpected metadata fields in {path.name}: {", ".join(unexpected_fields)}')
    organism, antigen = path.stem.rsplit('_', 1)
    metadata['organism'] = organism
    metadata['antigen'] = antigen
    return metadata


# def _download(repository: str, organism: str, antigen: str):
#     base_url = f'https://raw.githubusercontent.com/{repository}/refs/heads/master/reference_database/{organism}_{antigen}'
#     try:  # First try the metadata file, then genbank, these are REQUIRED
#         metadata = download(metadata_url := f'{base_url}.json', RESOURCES.data / f'{organism}_{antigen}.json')
#
#
