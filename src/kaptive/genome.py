from typing import Literal, IO, Union
from pathlib import Path
from itertools import chain
from random import Random

from . import RESOURCES
from .io import SeqFile
from .seq import Record, DNA
from .graph import Graph, Edge
from .utils import grouper


# Classes --------------------------------------------------------------------------------------------------------------
class GenomeError(Exception):
    pass


class Genome:
    """
    A class representing a single genome assembly in memory with contigs and potentially edges

    Attributes:
        id: The ID of the genome.
        contigs: A dictionary of contig IDs as keys and Record objects as values.
        edges: A list of Edges connecting the contigs.
    """
    def __init__(self, id_: str, contigs: dict[str: Record] = None, edges: list[Edge] = None):
        """Represents a single bacterial genome to be loaded into memory from a file"""
        self.id: str = id_
        self.contigs: dict[str: Record] = contigs or {}
        self.edges: list[Edge] = edges or []

    def __len__(self):
        return sum(len(i) for i in self.contigs.values())

    def __iter__(self):
        return iter(self.contigs.values())

    def __str__(self):
        return self.id

    def __getitem__(self, item: str) -> 'Record':
        return self.contigs[item]

    def __format__(self, __format_spec: Literal['fasta', 'fna', 'ffn', 'faa', 'bed', 'gfa'] = ''):
        if __format_spec == '':
            return self.__str__()
        elif __format_spec in {'fasta', 'fna', 'ffn', 'faa', 'bed'}:
            return ''.join(format(i, __format_spec) for i in self.contigs.values())
        elif __format_spec == 'gfa':
            return ''.join(format(i, __format_spec) for i in chain(self.contigs.values(), self.edges))
        else:
            raise NotImplementedError(f'Invalid format: {__format_spec}')

    @classmethod
    def from_file(cls, file: Union[str, Path, IO, SeqFile], annotations: Union[str, Path, SeqFile] = None):
        """
        Loads a genome from a file with optional annotations, e.g. a FASTA with a BED/GFF file.
        :param file: Path to file (str/Path) or a SeqFile instance.
        :param annotations: Optional annotations as a file path (str/Path) or SeqFile instance.
        :return: A Genome instance representing the genome in memory
        """
        file = SeqFile(file) if not isinstance(file, SeqFile) else file
        self = cls(file.id)
        for record in file:
            if isinstance(record, Record):
                self.contigs[record.id] = record
                if not file.format == 'gfa' and next((v for k, v in record.qualifiers if k == 'topology'),
                                                     'linear') == 'circular':
                    self.edges.append(Edge(record.id, record.id))  # Add self loop for circular genomes
                    # We assume a GFA file already has this edge but this may not be the case
            elif isinstance(record, Edge):
                self.edges.append(record)
        if annotations:
            if file.format not in {'fasta', 'gfa'}:
                raise GenomeError(f'Can only provide annotations to FASTA and GFA files, not {file.format}')
            annotations = SeqFile(annotations) if not isinstance(annotations, SeqFile) else annotations
            if not annotations.format in {'gff', 'bed'}:
                raise GenomeError(f'Annotations must be in GFF or BED format, not {annotations.format}')
            for contig_id, features in grouper(annotations, 'location.parent_id'):  # Sort by contig id
                if contig := self.contigs.get(contig_id):  # type: Record
                    contig.add_features(*features)
        return self

    @classmethod
    def random(
            cls, id_: str = None, genome: Record = None, rng: Random = None, n_contigs: int = None,
            min_contigs: int = 1, max_contigs: int = 1000, gc: float = 0.5, length: int = None, min_len: int = 10,
            max_len: int = 5000000
    ):
        """
        Generates a random genome assembly for testing purposes.

        :param id_: ID of the genome, if not provided a random one will be generated
        :param genome: Initial genome record; if not provided a random one will be generated
        :param rng: Random number generator.
        :param n_contigs: Number of contigs in the assembly. If not provided, a random number of contigs will be generated.
        :param min_contigs: Minimum number of contigs if n_contigs is not specified.
        :param max_contigs: Maximum number of contigs if n_contigs is not specified.
        :param gc: GC content of the genome.
        :param length: Total length of the genome. If not provided, a random length will be generated.
        :param min_len: Minimum length of the genome if length is not specified.
        :param max_len: Maximum length of the genome if length is not specified.
        :return: A Genome instance representing the random assembly.
        """
        if rng is None:
            rng = RESOURCES.rng
        contigs = {
            i.id: i for i in (genome or Record.random(
            None, DNA, rng, gc, length, min_len, max_len)).shred(rng, n_contigs or rng.randint(min_contigs, max_contigs))
        }
        return cls(id_ or DNA.hash(*(i.seq for i in contigs.values())), contigs)

    def annotated(self) -> bool:
        for contig in self.contigs.values():
            if contig.features: return True
        return False

    def as_graph(self):
        graph = Graph(*self.edges)
        for contig in self:
            graph.add_node(contig.id, dict(contig.qualifiers))
        return graph

    def as_feature_graph(self, genome_graph: Graph = None) -> Graph:
        """
        Returns a graph where features are nodes where adjacent features on the same contig and at the termini of connected
        contigs are connected.
        """
        genome_graph = genome_graph or self.as_graph()
        feature_graph = Graph()
        # 1. Add intra-contig edges (connecting adjacent features on the same contig)
        for contig_name, contig in self.contigs.items():
            for position, v in enumerate(contig.features):
                if position > 0:
                    u = contig.features[position - 1]
                    edge = Edge(u.id, v.id, {'u strand': u.location.strand, 'v strand': v.location.strand})
                    feature_graph.add_edge(edge)

            # 2. Add inter-contig edges by iterating through all assembly connections
            for edge in genome_graph.get_neighbors(contig):
                # Skip if either contig is missing or has no features
                if not (neighbor := self.contigs.get(edge.v)) or not neighbor.features:
                    continue
                # If the strand of current contig is 1, connect last feature, else connect the first
                u = contig.features[-1 if edge['u strand'] == 1 else 0]
                # If the strand of neighbor contig is 1, connect first feature, else connect the last
                v = neighbor.features[0 if edge['v strand'] == 1 else -1]
                # Add the new edge connecting the two features
                edge = Edge(u.id, v.id, {'u strand': u.location.strand, 'v strand': v.location.strand})
                feature_graph.add_edge(edge)

        return feature_graph

    def stitch(self, dag: Graph) -> Record:
        """
        Stitches together existing contigs using a directed acyclic graph and returns the new contig record.
        The old contigs will be updated by the new one.
        """
        assert len(dag) >= 2, ValueError('DAG must consist of at least 2 Edges')
        contigs = []
        for edge in dag:  # Extracting the contigs in this order asserts the graph is a DAG
            if not (u := self.contigs.pop(edge.u)):
                raise ValueError(f"Contig {edge.u} not found in genome.")
            if not (v := self.contigs.pop(edge.v)):
                raise ValueError(f"Contig {edge.v} not found in genome.")
            contigs.append(u)
            contigs.append(v)
        if not contigs:
            raise ValueError("No contigs to stitch.")

        new_contig = contigs[0]
        for contig in contigs[1:]:
            new_contig += contig
        # Add the new_contig to the genome's contigs
        self.contigs[new_contig.id] = new_contig

        # Update edges
        contigs = {contig.id: contig for contig in contigs}
        for edge in self.edges:
            if edge.u in contigs and edge.v in contigs:
                # If both ends of the edge are part of the stitched contigs, remove it
                # as it's now internal to the new_contig.
                pass
            elif edge.u in contigs:
                # If 'u' is part of the stitched contigs, update it to the new_contig ID
                edge.u = new_contig.id
            elif edge.v in contigs:
                # If 'v' is part of the stitched contigs, update it to the new_contig ID
                edge.v = new_contig.id
        # Remove duplicate edges that might have been created by updating
        self.edges = list(set(self.edges))

        return new_contig
