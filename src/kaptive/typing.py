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
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import chain
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import IO, Union, Generator, Callable, Literal
from warnings import warn

from kaptive import KaptiveWarning, require, RESOURCES
from kaptive.alignment import cull_filtered
from kaptive.db import Database
from kaptive.external import Minimap2, Minimap2AlignConfig, Minimap2IndexConfig
from kaptive.io import SeqFile, parse_cigar
from kaptive.genome import Genome
from kaptive.seq import Record, Feature, Qualifier, merge_locations
from kaptive.utils import Config, grouper


# Constants ------------------------------------------------------------------------------------------------------------
_ALIGNMENT_METRICS = Literal['score', 'n_matches', 'length', 'query_length']
_WEIGHT_METRICS = Literal[None, 'genes_found', 'genes_expected', 'proportion_genes_found', 'length', 'query_length']
_CONFIDENCE_LEVELS = Literal['Typeable', 'Untypeable']
_GENE_TYPES = Literal[
    'expected_inside_locus', 'unexpected_inside_locus', 'expected_outside_locus', 'unexpected_outside_locus', 'extra']
_ASSEMBLY_HEADER = ('Assembly\tBest match locus\tBest match type\tMatch confidence\tProblems\tIdentity\tCoverage\t'
                    'Length discrepancy\tExpected genes in locus\tExpected genes in locus, details\t'
                    'Missing expected genes\tOther genes in locus\tOther genes in locus, details\t'
                    'Expected genes outside locus\tExpected genes outside locus, details\t'
                    'Other genes outside locus\tOther genes outside locus, details\t'
                    'Truncated genes, details\tExtra genes, details\tComplete antigen\n')


# Classes --------------------------------------------------------------------------------------------------------------
class AssemblyTyperResult:
    """
    Holds the result of a typer run on a single genome
    """
    # TODO: Implement the possibility of calling multiple loci per genome
    def __init__(
            self, genome_id: str, db_id: str, best_match_id: str, phenotype_id: str = None,
            expected_genes: set[str] = None, identity: float = 0.0, coverage: float = 0.0,
            confidence: _CONFIDENCE_LEVELS = 'Untypeable', pieces: list[Record] = None,
            expected_inside_locus: list[Feature] = None, expected_outside_locus: list[Feature] = None,
            unexpected_inside_locus: list[Feature] = None, unexpected_outside_locus: list[Feature] = None,
            missing: set[str] = None, extra: list[Feature] = None, truncated: dict[str, Feature] = None,
            below_threshold: set[str] = None, partial: set[str] = None, n_contigs: int = 0, length_discrepancy: int = 0,
            units: set[str] = None, missing_units: set[str] = None
    ):
        self.genome_id: str = genome_id
        self.db_id: str = db_id
        self.best_match_id: str = best_match_id
        self.phenotype_id: str = phenotype_id or f'unknown ({best_match_id})'
        self.expected_genes: set[str] = expected_genes or set()
        self.identity: float = identity
        self.coverage: float = coverage
        self.confidence: _CONFIDENCE_LEVELS = confidence
        self.pieces: list[Record] = pieces or []  # Pieces of locus reconstructed from alignments
        self.expected_inside_locus: list[Feature] = expected_inside_locus or []
        self.expected_outside_locus: list[Feature] = expected_outside_locus or []
        self.unexpected_inside_locus: list[Feature] = unexpected_inside_locus or []
        self.unexpected_outside_locus: list[Feature] = unexpected_outside_locus or []
        self.missing: set[str] = missing or set()
        self.extra: list[Feature] = extra or []
        self.truncated: dict[str, Feature] = truncated or {}
        self.below_threshold: set[str] = below_threshold or set()
        self.partial: set[str] = partial or set()
        self.n_contigs: int = n_contigs
        self.length_discrepancy: int = length_discrepancy
        self.units: set[str] = units or set()
        self.missing_units: set[str] = missing_units or set()

    def problems(self) -> str:
        problems = []
        if len(self.pieces) > 1:
            problems.append(f'?{len(self.pieces)}')
        if self.missing:
            problems.append('-')
        if self.truncated:
            problems.append('!')
        if self.unexpected_inside_locus:
            problems.append('+')
        if self.below_threshold:
            problems.append('*')
        return ''.join(problems)

    def __repr__(self):
        return f"{self.genome_id} {self.best_match_id}"

    def __len__(self):
        return sum(len(i) for i in self.pieces) if self.pieces else 0

    def __format__(self, __format_spec: str = '') -> str:
        if __format_spec == '':
            return self.__str__()
        elif __format_spec in {'fasta', 'fna', 'ffn', 'faa', 'bed'}:
            return ''.join(format(i, __format_spec) for i in self.pieces)
        elif __format_spec == 'tsv':
            return '\t'.join(
                (
                    self.genome_id, self.best_match_id, self.phenotype_id, self.confidence, self.problems(),
                    f'{self.identity:.2%}', f'{self.coverage:.2%}',
                    f'{self.length_discrepancy} bp' if len(self.pieces) == 1 else 'n/a',
                    f"{(n_inside := len({i['name'] for i in self.expected_inside_locus}))} / "
                    f"{len(self.expected_genes)} ({n_inside / len(self.expected_genes):.2%})",
                    ';'.join(_feature_to_tsv(i, self) for i in self.expected_inside_locus),
                    ';'.join(self.missing),
                    f"{len(self.unexpected_inside_locus)}",
                    ';'.join(_feature_to_tsv(i, self) for i in self.unexpected_inside_locus),
                    f"{(n_outside := len({i['name'] for i in self.expected_outside_locus}))} / "
                    f"{len(self.expected_genes)} ({n_outside / len(self.expected_genes):.2%})",
                    ';'.join(_feature_to_tsv(i, self) for i in self.expected_outside_locus),
                    f"{len(self.unexpected_outside_locus)}",
                    ';'.join(_feature_to_tsv(i, self) for i in self.unexpected_outside_locus),
                    ';'.join(_feature_to_tsv(i, self) for i in self.truncated.values()),
                    ';'.join(_feature_to_tsv(i, self) for i in self.extra),
                    f'No ({";".join(self.missing_units)})' if self.missing_units else 'Yes'
                )
            ) + '\n'
        else:
            raise NotImplementedError(f'Invalid format: {__format_spec}')

    def __iter__(self):
        return iter(self.pieces)


class TyperWarning(KaptiveWarning):
    pass


class TyperError(Exception):
    pass


@dataclass
class AssemblyTyperConfig(Config):
    alignment_metric: _ALIGNMENT_METRICS = 'score'
    weight_metric: _WEIGHT_METRICS = 'proportion_genes_found'
    min_gene_cov_for_scoring: float = 0.5
    truncation_tolerance: float = 0.95
    stage_2_n_best: int = 2
    typeable_max_other_genes: int = 1
    typeable_prop_expected_genes: float = 0.5
    typeable_allow_below_threshold: bool = False


class AssemblyTyper:
    """
    Performs *in silico* serotyping on bacterial genome assemblies.
    """
    # @require('edlib', 'numpy')
    def __init__(self, db: Database, config: AssemblyTyperConfig = None, align_config: Minimap2AlignConfig = None,
                 index_config: Minimap2IndexConfig = None):
        try:
            from numpy import zeros, array
            from edlib import align
            self.align_proteins: Callable = partial(align, mode="NW", task="path")
        except ImportError as e:
            raise TyperError from e
        self.db: Database = db
        self.config: AssemblyTyperConfig = config or AssemblyTyperConfig()
        self.align_config: Minimap2AlignConfig = align_config or Minimap2AlignConfig()
        self.index_config: Minimap2IndexConfig = index_config or Minimap2IndexConfig()
        self.merge_tolerance: int = max(len(i) for i in self.db.loci.values())  # Len of the largest locus to merge pieces
        self.id_threshold: float = self.db.metadata.get('id_threshold', 80) / 100  # Turn into proportion
        self._locus2index: dict[str, int] = {}
        self._index2locus: list[str] = []
        self._scores = zeros((len(self.db.loci), 4))
        self._precomputed_weights = zeros(len(self.db.loci))
        for i, locus in enumerate(self.db.loci.values()):
            self._locus2index[locus.id] = i
            self._index2locus.append(locus.id)
            if self.config.weight_metric in {'genes_expected', 'proportion_genes_found'}:
                self._precomputed_weights[i] += len(locus.features)
        self._gene2index: dict[str, int] = {}
        self._index2gene: list[str] = []
        self._gene_scores = zeros((len(self.db.genes), 4))
        for i, gene in enumerate(self.db.genes.values()):
            self._gene2index[gene.id] = i
            self._index2gene.append(gene.id)

    def __repr__(self):
        return f"AssemblyTyper({self.db.id})"

    def _pipeline(self, genome: Union[str, Path, IO, SeqFile, Genome]) -> Union['AssemblyTyperResult', None]:
        """
        Performs *in silico* serotyping on a single bacterial genome genome
        :param genome: A bacterial ``Genome`` instance or file in FASTA, GFA or Genbank format
        :return: ``AssemblyTypingResult`` object
        """
        if not isinstance(genome, Genome):
            try:
                genome = Genome.from_file(genome)
            except Exception as e:
                raise TyperError(f"Could not parse {genome} as Genome") from e
        else:
            genome = deepcopy(genome)  # Create a copy of the genome instance to leave the original unmodified

        aligner = Minimap2([f'{self.db:ffn}'], self.align_config, self.index_config)
        gene_scores = self._gene_scores.copy()
        alignments_per_gene = {}
        for gene, alns in grouper(aligner.align([f'{genome:fasta}']), 'location.parent_id'):
            alns = [i.flip() for i in alns]
            best_aln = max(alns, key=lambda x: x.score)  # type: Alignment
            gene_scores[self._gene2index[best_aln.query]] = [best_aln.score, best_aln.identity, best_aln.query_coverage, best_aln.length]
            alignments_per_gene[gene] = alns

        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=len(self.db.loci), random_state=0)
        gmm.fit(gene_scores)
        probabilities = gmm.predict_proba(gene_scores)
        from numpy import argmax
        # Assign each alignment to the most likely locus
        assignments = argmax(probabilities, axis=1)

        locus_assignments = {k: self._index2locus[v] for k, v in zip(self.db.genes.keys(), assignments)}


    def __call__(self, genome: Union[str, Path, IO, SeqFile, Genome]) -> Union['AssemblyTyperResult', None]:
        """
        Performs *in silico* serotyping on a single bacterial genome genome
        :param genome: A bacterial ``Genome`` instance or file in FASTA, GFA or Genbank format
        :return: ``AssemblyTypingResult`` object
        """
        if not isinstance(genome, Genome):
            try:
                genome = Genome.from_file(genome)
            except Exception as e:
                raise TyperError(f"Could not parse {genome} as Genome") from e
        else:
            genome = deepcopy(genome)  # Create a copy of the genome instance to leave the original unmodified

        # Enter the aligner context, this builds the index from the genome, we will use it twice
        with Minimap2(genome, self.align_config, self.index_config) as aligner:
            # Stage 1: align genes -------------------------------------------------------------------------------------
            scores, gene_alignments_by_contig, scored = dict.fromkeys(self.db.loci.keys(), 0), {}, False
            weights = self.precomputed_weights.copy()
            gene_queries = chain(self.db.genes.values(), self.db.extra_genes.values())  # Prepare queries
            for gene, alns in grouper(aligner.align(gene_queries), 'query'):  # Group alignments by query gene
                best_alignment = max((alns := list(alns)), key=attrgetter('score'))  # Add the best alignment for extra genes
                if gene.startswith("Extra_genes"):
                    gene_alignments_by_contig.setdefault(best_alignment.target, []).append(best_alignment)
                else:
                    for alignment in alns:  # Keep all alignments
                        gene_alignments_by_contig.setdefault(alignment.target, []).append(alignment)
                    if best_alignment.query_coverage >= self.config.min_gene_cov_for_scoring:
                        locus = best_alignment.query.split('_', 1)[0]
                        scores[locus] += attrgetter(self.config.alignment_metric)(best_alignment)
                        if self.config.weight_metric:
                            if self.config.weight_metric == 'genes_found':
                                weights[locus] += 1  # Weights are set to 0
                            elif self.config.weight_metric == 'proportion_genes_found':
                                weights[locus] -= 1  # Weights are set to the number of genes expected
                            elif self.config.weight_metric in {'length', 'query_length'}:
                                weights[locus] += attrgetter(self.config.weight_metric)(best_alignment)
                        scored = True
            if not scored:
                return warn(f'No gene alignments sufficient for typing {genome}\nHave you used the appropriate '
                              f'database for your species?', TyperWarning)  # Warns and returns None

            # Stage 2: align loci --------------------------------------------------------------------------------------
            top_loci = (self.db.loci[i[0]] for i in sorted(((k, _div(v, weights[k])) for k, v in scores.items()),
                                                           key=itemgetter(1), reverse=True)[0:self.config.stage_2_n_best])
            scores2 = []  # Collect 2nd scores, we can't re-use scores as it is used in the generator above
            for locus, locus_alignments in grouper(aligner.align(top_loci), 'query'):
                scores2.append((locus, a := list(locus_alignments), sum(map(attrgetter('score'), a))))

        # Stage 3: Reconstruct locus -----------------------------------------------------------------------------------
        # We no longer need the aligner so exit the context here which cleans up the genome index
        best_match, locus_alignments, _ = max(scores2, key=itemgetter(2))
        pieces = {c: list(merge_locations(a, self.merge_tolerance)) for c, a in grouper(locus_alignments, 'target')}
        best_match = self.db.loci[best_match]
        expected_genes, found_genes = {i.id for i in best_match}, set()
        result = AssemblyTyperResult(genome.id, self.db.id, best_match.id, best_match.desc, expected_genes, n_contigs=len(pieces))
        current_phenotype = {'truncated': set(), 'extra': set()}

        for contig, genes in gene_alignments_by_contig.items():
            contig_pieces, features = pieces.get((contig := genome[contig]).id, []), []

            # Stage 4: Assess genes ------------------------------------------------------------------------------------
            for gene_aln in cull_filtered(lambda i: i.query in expected_genes, genes):
                if gene_aln.query.startswith('Extra'):
                    gene_type = 'extra'
                elif gene_aln.query in expected_genes:
                    gene_type = 'expected'
                    found_genes.add(gene_aln.query)
                else:
                    gene_type = 'unexpected'

                # Get the reference gene and potential piece, then convert the gene to a feature
                ref = (self.db.extra_genes if gene_type == 'extra' else self.db.genes)[gene_aln.query]
                # Convert alignment to a Feature object
                gene = gene_aln.as_feature('CDS')  # It becomes a CDS as it represents a coding sequence
                # Create unique gene identifier for fasta headers and makes the gene unique if there are multiple pieces
                gene.id = (gene_id := f'{gene_aln.query}|{genome}|{contig}|{gene.location}')
                # Find the corresponding locus piece the gene is on using location.__contains__() method
                if next((i for i in contig_pieces if gene.location.overlap(i)), None):
                    features.append(gene)  # Add feature to record later on,
                    if not gene_type == 'extra':
                        gene_type += '_inside_locus'
                elif not gene_type == 'extra':
                    gene_type += '_outside_locus'

                if gene.partial() and gene_type == 'unexpected_outside_locus':   # Partial outside locus
                    continue # We skip these genes as they are probably homolog fragments

                # Determine the correct translation frame based on the alignment information
                # predicted_frame = (-gene_aln.query_start) % 3 if gene_aln.location.strand == 1 else (
                #         (gene_aln.query_length - gene_aln.query_end) % 3)

                # gene_translation = gene.translate(parent=contig, store_seq=True, frame=frame)
                translations = {
                    0: gene.translate(parent=contig, store_seq=True, frame=0),
                    1: gene.translate(parent=contig, store_seq=True, frame=1),
                    2: gene.translate(parent=contig, store_seq=True, frame=2)
                }
                frame, gene_translation = max(translations.items(), key=lambda i: len(i[1]))
                # if frame != predicted_frame:
                #     print(f'Longest translation for {gene} resulted from {frame=} not {predicted_frame=}')

                ref_translation = ref.translate()  # Get reference protein sequence for comparison
                protein_identity = 1.0   # If DNA identity is 100%, protein identity is also 100%.
                if gene_aln.identity < 1:   # Otherwise, perform a protein alignment
                    protein_identity = self._align_proteins(str(gene_translation), str(ref_translation))
                protein_coverage = len(gene_translation) / len(ref_translation)

                if not gene.partial():
                    if protein_coverage < self.config.truncation_tolerance:  # This protein is truncated
                        if gene_type == 'extra':
                            continue  # We don't care about truncated extra genes
                        elif gene_type == 'expected_inside_locus':
                            result.truncated[gene_id] = gene
                            current_phenotype['truncated'].add(gene_aln.query)
                    elif protein_identity < self.id_threshold:
                        if gene_type == 'expected_inside_locus':
                            result.below_threshold.add(gene_id)  # Below identity threshold
                        else:
                            continue  # We don't care about truncated unexpected genes
                else:
                    result.partial.add(gene_id)

                if gene_type == 'extra':
                    current_phenotype['extra'].add(gene_aln.query)

                gene.qualifiers += (Qualifier('protein_identity', protein_identity),
                                    Qualifier('protein_coverage', protein_coverage))

                # Add gene to result
                getattr(result, gene_type).append(gene)

            # Stage 5: Finalise piece ----------------------------------------------------------------------------------
            contig.add_features(*features)  # Add genes inside locus to contig
            result.pieces += (contig[piece] for piece in contig_pieces)  # Add locus piece records to result

        # Stage 6: Finalise result -------------------------------------------------------------------------------------
        result.missing = expected_genes - found_genes  # Determine missing genes
        result.length_discrepancy = len(result) - len(best_match)  # Determine length discrepancy
        if result.expected_inside_locus:
            for gene in result.expected_inside_locus:
                result.identity += gene['protein_identity']
                result.coverage += len(gene)
            result.identity = min(result.identity / len(result.expected_inside_locus), 1)
            result.coverage = min(result.coverage / sum(len(i) for i in best_match.features), 1)

        # Calculate missing antigenic units
        for gene in expected_genes:
            if (unit := self.db.antigenic_units.get(gene)) and gene not in current_phenotype['truncated']:
                result.units.add(unit)
        result.missing_units = self.db.expected_units[result.best_match_id] - result.units

        # Calculate phenotype
        if phenotypes := self.db.phenotypes.get(result.best_match_id):
            if any(v for v in current_phenotype.values()):
                phenotype_scores = []
                for phenotype, states in phenotypes.items():
                    if score := sum(len(current_phenotype.get(state, set()) & genes) for state, genes in states.items()):
                        phenotype_scores.append((phenotype, score))
                if phenotype_scores:
                    result.phenotype_id = max(phenotype_scores, key=itemgetter(1))[0]

        # Calculate confidence - default is "Untypeable" so we are trying to find evidence it is "Typeable"
        if self.config.typeable_allow_below_threshold or not result.below_threshold:
            if len(result.pieces) == 1:  # Locus is in a single piece
                if not result.missing and not result.unexpected_inside_locus:  # No missing or unexpected genes in locus
                    result.confidence = "Typeable"
            elif (len({i['gene'] for i in result.unexpected_inside_locus}) <= self.config.typeable_max_other_genes and
                  len(found_genes) / len(expected_genes) >= self.config.typeable_prop_expected_genes):
                result.confidence = "Typeable"

        return result

    def _align_proteins(self, gene_translation: str, ref_translation: str) -> float:
        """Helper function to align two protein sequences using the protein aligner attribute"""
        protein_alignment_result = self.align_proteins(gene_translation, ref_translation)
        if protein_alignment_result and protein_alignment_result['editDistance'] != -1:
            cigar, matches, alignment_len = protein_alignment_result['cigar'], 0, 0
            # edlib CIGAR uses =, X, I, D. parse_cigar from alignment module handles this.
            for op, n, _, _, aln_len in parse_cigar(cigar):
                if op == '=':  # Match
                    matches += n
                alignment_len += aln_len
            return (matches / alignment_len) if alignment_len > 0 else 0.0
        return 0.0

    def map(self, *genomes: Union[str, Path, IO, SeqFile, Genome], executor: 'concurrent.futures.Executor' = None
            ) -> Generator[AssemblyTyperResult, None, None]:
        """
        Runs the pipeline on input genomes, using an optional executor

        Parameters:
            genomes: One or more input genomes
            executor: Optional concurrent.futures.Executor instance for mapping concurrently

        Yields:
            AssemblyTyperResult per input genome or None

        """
        yield from (executor.map if executor else map)(self, genomes)


# Functions ------------------------------------------------------------------------------------------------------------
def _feature_to_tsv(feature: Feature, result: AssemblyTyperResult) -> str:
    """Helper function to format gene features for TSV output"""
    assert isinstance(feature, Feature), TyperError(feature)
    return (f'{feature["name"]},{feature["protein_identity"]:.2%},{feature["protein_coverage"]:.2%}'
            f'{",partial" if feature.id in result.partial else ""}'
            f'{",truncated" if feature.id in result.truncated else ""}'
            f'{",below_id_threshold" if feature.id in result.below_threshold else ""}')


def _div(a: Union[int, float], b: Union[int, float, None]) -> float:
    """Avoid a ZeroDivisionError"""
    return a if not b else a / b


def main():
    # genome = Genome.from_file('../KL2_pw_genomes/SAMN06438497.fasta')
    # genome = Genome.from_file('../KL2_pw_genomes/SAMEA113605912.fasta')
    # genome = Genome.from_file('../KL2_pw_genomes/SAMEA10303181.fasta')
    genome = Genome.from_file('/Users/tsta0015/Programming/Kaptive4/kaptive/test/data/ERR4920433.gfa')
    self = AssemblyTyper(Database.from_package('kpsc_k'))
    # result = self(genome)
    # k_typer = AssemblyTyper(Database.from_package('kpsc_k'))
    # with open('pw_KL2_K.tsv', 'wt') as k_tsv:
    #     k_tsv.write(_ASSEMBLY_HEADER)
    #     from concurrent.futures import ThreadPoolExecutor
    #     with ThreadPoolExecutor(8) as executor:
    #         for result in k_typer.map(*Path('../KL2_pw_genomes').iterdir(), executor=executor):
    #             if result:
    #                 k_tsv.write(f'{result:tsv}')

