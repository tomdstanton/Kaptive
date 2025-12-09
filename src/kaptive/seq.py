"""
Module containing data structures for representing bacterial sequence components, similar to Biopython.
"""
from copy import deepcopy
from operator import attrgetter
from typing import Literal, Iterator, Union, Any, Generator, Iterable
from copy import copy
from random import Random
from re import compile as regex
from hashlib import new

from . import RESOURCES, KaptiveWarning
from .graph import Edge, GraphError
from .utils import encode_strand, decode_strand

# Constants ------------------------------------------------------------------------------------------------------------
_TYPE2TAG = {float: 'f', int: 'i', str: 'Z'}  # See: https://github.com/GFA-spec/GFA-spec/blob/master/GFA1.md#optional-fields
# _PROMOTER_REGEX = regex("r(?P<minus_35>TTG[AC][AC]A).{15,21}(?P<minus_10>TATA[AT]T)")
_PROMOTER_REGEX = regex(r"(?P<minus_35>TT[GCA][ATGC]{2}[A]).{15,21}(?P<minus_10>TA[ATGC]{2}A[AT])")
_TRANSLATION_TABLES: dict[int, dict[str]] = {
    11: {
        'translation': 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG',
         'starts': {'ATG', 'GTG', 'TTG', 'ATT', 'ATC', 'CTG'},
         'stops': {'TAA', 'TAG', 'TGA'}
    }
}

# Classes --------------------------------------------------------------------------------------------------------------
class AlphabetError(Exception):
    pass


class Alphabet:
    """
    A class to represent an alphabet of symbols.
    """
    def __init__(self, name: str, symbols: str, hash_algorithm: str = 'sha1'):
        self.symbols = symbols
        self.set: frozenset[str] = frozenset(symbols)
        if len(self.set) < len(self.symbols):
            raise AlphabetError(f'Alphabet symbols "{symbols}" are not unique')
        self.hash_algorithm = hash_algorithm

    def __len__(self):
        return len(self.symbols)

    def __repr__(self):
        return self.symbols

    def __hash__(self):
        return hash(self.set)

    def __eq__(self, other):
        if isinstance(other, Alphabet):
            return self.symbols == other.symbols
        return False

    def __contains__(self, item: Union[str, set[str]]) -> bool:
        if isinstance(item, str):
            return self.set.intersection(item)
        elif isinstance(item, set):
            return set(item) <= self.set
        else:
            raise ValueError(f"Unexpected item type {type(item)}")

    def __iter__(self):
        return iter(self.symbols)

    def __getitem__(self, item):
        return self.symbols[item]

    def hash(self, *sequences: 'Seq') -> str:
        """
        Hashes sequences using a hash algorithm, used for generating locus tags
        :param sequences: Seq objects to hash
        :return: Hash string
        """
        h = new(self.hash_algorithm, usedforsecurity=False)
        for seq in sequences:
            if seq.alphabet != self:
                raise AlphabetError(f"Seq alphabet is {seq.alphabet}, not {self}")
            h.update(bytes(seq))
        return h.hexdigest()


class AminoAcidAlphabet(Alphabet):
    """
    Child-class of Alphabet to represent the IUPAC unambiguous Amino Acid symbols ACDEFGHIKLMNPQRSTVWY,
    and extra methods unique to Amino Acids.
    """

    def __init__(self, name, stop_symbol: str = '*'):
        super().__init__(name, 'ACDEFGHIKLMNPQRSTVWY', 'md5')
        self.stop_symbol = stop_symbol


class TranslationError(AlphabetError):
    pass


class TranslationWarning(KaptiveWarning):
    pass


class NucleotideAlphabet(Alphabet):
    """
    Child-class of Alphabet to represent the IUPAC unambiguous DNA symbols TCAG, and extra methods unique to DNA.
    """

    def __init__(self, name, translation_table: int = 11):
        if not (t := _TRANSLATION_TABLES.get(translation_table)):
            raise AlphabetError(f'Unknown translation table {translation_table}')
        super().__init__(name, 'TCAG')
        self.translation_table: int = translation_table
        self._complement: dict[int, int] = str.maketrans('TCAG', 'AGTC')
        self._codons: dict[str, str] = dict(zip(
            (a + b + c for a in self.symbols for b in self.symbols for c in self.symbols), t['translation']))
        self.start_codons = t['starts']
        self.stop_codons = t['stops']

    def __repr__(self):
        return f"{self.symbols}; {self.translation_table=}"

    def translate(self, seq: str, to_stop: bool = True, frame: Literal[0, 1, 2] = 0, gap_character: str = None) -> str:
        """
        Translates the sequence to amino acid
        :param seq: The nucleotide sequence to translate
        :param to_stop: Boolean to stop translation at the first stop codon
        :param frame: Zero-based frame to begin translation from, must be one of 0, 1 or 2
        :param gap_character: The gap character if performing a gapped-translation
        :returns: A tuple of the translated sequence and the alphabet name
        """
        if len(seq := seq[frame:]) < 3:
            raise TranslationError(f'Cannot translate sequence of length {len(seq)}')
        protein = []
        for i in range(0, len(seq), 3):  # Iterate over seq codons (chunks of 3)
            if len(codon := seq[i:i + 3]) < 3:
                break  # We can't translate chunks less than 3 (not codons) so break here
            if gap_character and gap_character in codon:
                protein.append(gap_character)
            else:
                protein.append(residue := self._codons.get(codon, Protein.stop_symbol))
                if to_stop and residue in Protein.stop_symbol:
                    break  # Break if to_stop == True
        if (protein := ''.join(protein)) == Protein.stop_symbol:
            warn(f'Translated to a single stop symbol: {seq}', TranslationWarning)
        return protein

    def complement(self, seq: str):
        return seq.translate(self._complement)


# Supported alphabets --------------------------------------------------------------------------------------------------
DNA = NucleotideAlphabet('DNA')
Protein = AminoAcidAlphabet('Protein')


# Classes --------------------------------------------------------------------------------------------------------------
class LocationError(Exception):
    pass


class Location:
    """
    Represents a region on a sequence.

    :param start: Start coordinate (0-based)
    :param end: End coordinate (0-based)
    :param strand: Strand (1 or -1)
    :param partial_start: Boolean indicating if the start is partial
    :param partial_end: Boolean indicating if the end is partial
    :param parent_id: Reference sequence name
    """

    def __init__(self, start: int, end: int, strand: Literal[1, -1] = 1, partial_start: bool = False,
                 partial_end: bool = False, parent_id: str = 'unknown'):
        self.start: int = start
        self.end: int = end
        self.strand: Literal[1, -1] = encode_strand(strand)
        self.partial_start: bool = partial_start
        self.partial_end: bool = partial_end
        self.parent_id: str = parent_id

    @classmethod
    def from_slice(cls, s: slice) -> 'Location':
        if s.start >= s.stop:
            return cls(s.stop, s.start, -1)
        return cls(s.start, s.stop, 1)

    def __hash__(self):
        return hash((self.start, self.end, self.strand, self.parent_id))

    def __str__(self):
        return f"{self.parent_id}_{self.start}_{self.end}_{decode_strand(self.strand)}"

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        return iter((self.start, self.end, self.strand))

    def __contains__(self, item: Union['Location', 'HasLocation', int, float, slice]):
        if isinstance(item, slice):
            return self.start <= item.start and self.end >= item.stop
        elif isinstance(item, (Location, HasLocation)):
            if isinstance(item, HasLocation):
                item = item.location
            return self.start <= item.start and self.end >= item.end
        elif isinstance(item, (int, float)):
            return self.start <= item <= self.end
        else:
            raise TypeError(item)

    def __add__(self, other: Union['Location', 'HasLocation']) -> 'Location':
        if isinstance(other, HasLocation):
            other = other.location
        if isinstance(other, Location):
            return Location(min(self.start, other.start), max(self.end, other.end), self.strand)
        else:
            raise TypeError(other)

    def __radd__(self, other: Union['Location', 'HasLocation']) -> 'Location':
        if isinstance(other, HasLocation):
            other = other.location
        if isinstance(other, Location):
            return other.__add__(self)
        else:
            raise TypeError(other)

    def __iadd__(self, other: Union['Location', 'HasLocation']):
        if isinstance(other, HasLocation):
            other = other.location
        if isinstance(other, Location):
            self.start = min(self.start, other.start)
            self.end = max(self.end, other.end)
            return self
        else:
            raise TypeError(other)

    def __format__(self, __format_spec: Literal['tsv'] = '') -> str:
        if __format_spec == '':
            return self.__str__()
        elif __format_spec == 'tsv':
            return f'{self.parent_id}\t{self.start}\t{self.end}\t{self.strand}'
        else:
            raise NotImplementedError(f'Invalid format: {__format_spec}')

    def __delitem__(self, item: Union[slice, int, 'Location', 'HasLocation']):
        if isinstance(item, Location):
            pass
        elif isinstance(item, HasLocation):
            item = item.location
        elif isinstance(item, slice):
            item = Location.from_slice(item)
        elif isinstance(item, int):
            item = Location.from_slice(slice(item, item))
        else:
            raise TypeError(item)
        self.start = max(self.start, item.start)
        self.end = min(self.end, item.end)

    def overlap(self, other: Union['Location', 'HasLocation']) -> int:
        """
        Returns the length of the overlap between two locations
        :param other: Location or HasLocation object
        """
        if isinstance(other, HasLocation):
            other = other.location
        if isinstance(other, Location):
            return max(0, min(self.end, other.end) - max(self.start, other.start))
        else:
            raise TypeError(other)

    def extract(self, parent: Union['Seq', 'Record', 'Feature']) -> 'Seq':
        """
        Extracts a sequence from a parent object based on the location.

        :param parent: Parent object (``Seq``, ``Record``, ``Feature``) or dictionary of parent objects
        :return: Seq object
        """
        if not parent:
            raise LocationError(f"{parent=} is not truthy, does it have an empty sequence?")
        if self.parent_id and not isinstance(parent, Seq) and parent.id != self.parent_id:
            raise LocationError(f'{self.parent_id=} does not match {parent.id=}')
        return parent.seq[self] if not isinstance(parent, Seq) else parent[self]

    def shift(self, by: int):
        return Location(self.start + by, self.end + by, self.strand, self.partial_start, self.partial_end, self.parent_id)

    def reverse_complement(self, parent_length: int) -> 'Location':
        return Location(
            parent_length - self.end, parent_length - self.start, -self.strand,
            self.partial_end, self.partial_start, self.parent_id
        )

    @classmethod
    def random(cls, rng: Random = None, length: int = None, min_len: int = 1, max_len: int = 10000,
               min_start: int = 0, max_start: int = 1000000):
        if rng is None:
            rng = RESOURCES.rng
        if not length:
            length = rng.randint(min_len, max_len)
        start = rng.randint(min_start, max_start - length)
        return cls(start, start + length, rng.choice([1, -1]))


class HasLocation:
    """
    Base class for objects with a location attribute
    """

    def __init__(self, location: Location):
        self.location = location

    def __len__(self) -> int:
        return self.location.end - self.location.start

    def overlap(self, other: Union[Location, 'HasLocation']) -> int:
        """
        Returns the length of the overlap between two locations
        """
        return self.location.overlap(other)

    def __contains__(self, item: Union[Location, 'HasLocation', int, float]) -> bool:
        """
        Returns True if the object contains the item, False otherwise
        """
        return self.location.__contains__(item)


class SeqError(Exception):
    pass


class Seq:
    """
    An abstract base class for biological sequences.

    This class acts as a factory. When you call Seq("some_string"), it will
    inspect the characters and return an instance of a more specific
    subclass (e.g., NucleotideSeq or ProteinSeq).

    It cannot be instantiated directly.
    """

    def __new__(cls, seq: str = None, alphabet: Alphabet = None):
        """
        Creates a new sequence object, guessing the alphabet if not provided.
        This is the factory part of the class.
        """
        # If seq is not provided, it's likely being called by deepcopy.
        # In this case, just create a bare object and let deepcopy handle state.
        if seq is None:
            return super().__new__(cls)
        # If this method is called by a subclass (e.g., NucleotideSeq("atgc")),
        # don't re-run the factory logic. Just create a new instance of that subclass.
        if cls is not Seq:
            return super().__new__(cls)

        # --- Factory logic: only runs when calling Seq(...) directly ---
        if not isinstance(seq, str):
            raise TypeError(f"Sequence data must be a string, not {type(seq).__name__}")

        # Process the string once for alphabet detection and construction.
        processed_seq = seq.strip().upper()
        if not processed_seq:
            raise SeqError("Seq must be >=1 character(s)")

        if alphabet is None:  # Guess the alphabet based on the symbols
            # Check for DNA, allowing for 'N' which is a valid IUPAC nucleotide
            if set(processed_seq.replace('N', '')) <= DNA.set:
                target_cls = NucleotideSeq
            # Check for Protein
            elif (symbols := set(processed_seq)) <= Protein.set:
                target_cls = ProteinSeq
            else:
                raise SeqError(f"Could not automatically determine alphabet for sequence with symbols: {symbols}")
        else:  # If alphabet is specified, choose the correct class
            if alphabet is DNA:
                target_cls = NucleotideSeq
            elif alphabet is Protein:
                target_cls = ProteinSeq
            else:  # For other alphabets, use the base object creator and let __init__ handle it.
                return super().__new__(cls)  # Let __init__ handle this, which will raise a TypeError

        # Delegate instantiation to the determined class's constructor with the processed sequence
        return target_cls(processed_seq)

    def __init__(self, seq: str, alphabet: Alphabet = None):
        """
        Initializes a sequence object. This method is called after __new__.
        """
        if type(self) is Seq:  # Make the base class abstract; it cannot be instantiated directly.
            raise TypeError("Seq is an abstract base class and cannot be instantiated directly. "
                            "Use NucleotideSeq, ProteinSeq, or call Seq() to auto-detect.")

        if not isinstance(seq, str):
            raise TypeError(f"Sequence data must be a string, not {type(seq).__name__}")
        # This processing is necessary for direct instantiation of subclasses (e.g., NucleotideSeq(" atgc ")).
        # When called from the Seq() factory, `seq` is already processed,
        # making this a harmless and fast no-op.
        self._seq = seq.strip().upper()
        self.alphabet = alphabet

    def __repr__(self):
        return f"{type(self).__name__}('{self._seq if len(self._seq) < 13 else f'{self._seq[:5]}...{self._seq[-5:]}'}')"

    def __len__(self):
        return len(self._seq)

    def __hash__(self) -> int:
        return hash(self._seq)

    def __eq__(self, other):
        if isinstance(other, Seq):
            return self._seq == other._seq and self.alphabet == other.alphabet
        elif isinstance(other, str):
            return self._seq == other.upper()
        return False

    def __str__(self):
        return self._seq

    def __bytes__(self):
        return self._seq.encode()

    def __iter__(self):
        return iter(self._seq)

    def __contains__(self, item):
        if isinstance(other, Seq):
            return self._seq in other._seq
        elif isinstance(item, str):
            return item.upper() in self._seq
        return False

    def __reversed__(self) -> 'Seq':
        # Use type(self) to ensure the returned object is of the same class (e.g., NucleotideSeq)
        return type(self)(self._seq[::-1])

    def __add__(self, other: Union[str, 'Seq']) -> 'Seq':
        if isinstance(other, Seq):
            if self.alphabet != other.alphabet:
                raise SeqError('Both sequences need to be of the same alphabet')
            return type(self)(self._seq + str(other))
        elif isinstance(other, str):
            # Validate the string's characters against the current alphabet without creating a new Seq object
            processed_other = other.strip().upper()
            if not set(processed_other) <= self.alphabet.set:
                raise SeqError(f"String '{other[:20]}...' contains characters not in the {self.alphabet} alphabet.")
            return type(self)(self._seq + processed_other)
        else:
            raise TypeError(other)

    def __radd__(self, other: Union[str, 'Seq']) -> 'Seq':
        if isinstance(other, str):
            # Let __add__ handle the logic
            return self.__add__(other)
        # The __add__ method on the other object will handle Seq + Seq
        return NotImplemented

    def __iadd__(self, other: Union[str, 'Seq']):
        if isinstance(other, Seq):
            if self.alphabet != other.alphabet:
                raise SeqError('Both sequences need to be of the same alphabet')
            self._seq += str(other)
        elif isinstance(other, str):
            temp_seq = Seq(other)
            if self.alphabet != temp_seq.alphabet:
                raise SeqError(
                    f'Cannot add string with alphabet {temp_seq.alphabet} to sequence with alphabet {self.alphabet}')
            self._seq += str(temp_seq)
        else:
            raise TypeError(other)
        return self

    def __getitem__(self, item: Union[slice, int, Location, HasLocation]) -> 'Seq':
        """Quick method of slicing the sequence and reverse complementing if necessary"""
        if isinstance(item, (slice, int)):
            return type(self)(self._seq[item])
        elif isinstance(item, (Location, HasLocation)):
            if isinstance(item, HasLocation):
                item = item.location
            # Use the class's constructor to create a new instance
            new_seq_obj = type(self)(self._seq[item.start:item.end])
            return new_seq_obj if (item.strand == 1) else new_seq_obj.reverse_complement()
        else:
            raise TypeError(item)

    def __delitem__(self, item: Union[slice, int, Location, HasLocation]):
        if isinstance(item, HasLocation):
            start, end = item.location.start, item.location.end
        elif isinstance(item, Location):
            start, end = item.start, item.end
        elif isinstance(item, slice):
            start, end, _ = item.indices(len(self._seq))
        elif isinstance(item, int):
            if item < 0:
                item += len(self._seq)
            start, end = item, item + 1
        else:
            raise TypeError(item)
        if start >= end or start >= len(self._seq):
            return  # Nothing to delete

        self._seq = self._seq[:start] + self._seq[end:]

    def __call__(self) -> str:
        return self._seq

    def hash(self):
        return self.alphabet.hash(self)

    @classmethod
    def random(cls, alphabet: Alphabet, rng: Random = None, length: int = None, min_len: int = 5, max_len: int = 5000) -> 'Seq':
        if rng is None:
            rng = RESOURCES.rng
        return cls(''.join(rng.choice(list(alphabet)) for _ in range(length or rng.randint(min_len, max_len))), alphabet)


class ProteinSeq(Seq):
    def __init__(self, seq: str):
        super().__init__(seq, Protein)

    @classmethod
    def random(cls, rng: Random = None, length: int = None, min_len: int = 5, max_len: int = 5000) -> 'ProteinSeq':
        if rng is None:
            rng = RESOURCES.rng
        return cls(''.join(rng.choice(list(Protein)) for _ in range(length or rng.randint(min_len, max_len))))

    def reverse_complement(self):
        raise NotImplementedError("Protein sequences cannot be reverse-complemented.")


class NucleotideSeq(Seq):
    def __init__(self, seq: str):
        super().__init__(seq, DNA)

    @classmethod
    def random(cls, rng: Random = None, gc: float = 0.5, length: int = None, min_len: int = 10,
               max_len: int = 5000000) -> 'NucleotideSeq':
        if rng is None:
            rng = RESOURCES.rng
        at = (1 - gc) / 2
        gc_comp = gc / 2
        return cls(''.join(
            rng.choices(DNA.symbols, weights=[at, gc_comp, at, gc_comp], k=length or rng.randint(min_len, max_len))))

    def reverse_complement(self) -> 'NucleotideSeq':
        """Reverse complements the sequence if it is a nucleotide sequence"""
        return NucleotideSeq(DNA.complement(self._seq)[::-1])

    def translate(self, *args, **kwargs) -> 'ProteinSeq':
        """
        Translates the sequence to amino acid if it is a nucleotide sequence
        """
        return ProteinSeq(DNA.translate(self._seq, *args, **kwargs))

    def find_promoters(self, reverse_complement: bool = False) -> Generator['Location', None, None]:
        """
        Finds promoters within the sequence.

        :return: Generator of Location objects representing the promoters
        """
        for match in _PROMOTER_REGEX.finditer(self._seq):
            yield Location(match.start(), match.end(), 1)
        if reverse_complement:
            rc_seq = self.reverse_complement()
            for match in _PROMOTER_REGEX.finditer(str(rc_seq)):
                # Adjust location back to the forward strand's coordinate system
                yield Location(len(self) - match.end(), len(self) - match.start(), -1)


class Qualifier:
    """
    Class to represent a qualifier as a key and value

    Attributes:
        key: str
        value: Any
    """

    def __init__(self, key: str, value: Any = None):
        self.key = key
        self.value = value if value != '' or value is not None else ''  # We still want zeroes

    def __repr__(self):
        return f"{self.key}={self.value}" if self.value != '' or self.value is not None else self.key

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter((self.key, self.value))

    def __str__(self):
        return f"{self.key}:{_TYPE2TAG[type(self.value)]}:{self.value}"

    def __eq__(self, other):
        if isinstance(other, Qualifier):
            return self.key == other.key and self.value == other.value
        return False


class Record:
    """
    Represents a sequence record.

    Attributes:
        id: str
        seq: The sequence of the record.
        desc: A description of the record.
        qualifiers: A list of Qualifier objects providing additional information about the record.
        features: A list of Feature objects representing features on the sequence.
    """

    def __init__(self, seq: Union[Seq, str], id_: str = None, desc: str = None, qualifiers: list[Qualifier] = None,
                 features: list['Feature'] = None):
        self.seq = seq if isinstance(seq, Seq) else Seq(seq)
        self.id = id_ or self.seq.hash()
        self.desc = desc or ''
        self.qualifiers = deepcopy(qualifiers) if qualifiers is not None else []
        self.features = deepcopy(features) if features is not None else []  # Deep copy features for full independence

    def __getitem__(self, item: Union[slice, HasLocation, Location]) -> 'Record':
        if isinstance(item, HasLocation):  # Convert HasLocation to Location
            item = item.location
        elif isinstance(item, slice):  # Convert slice to location
            item = Location(item.start or 0, item.stop or len(self), 1)
        elif not isinstance(item, Location):
            raise TypeError(f"Item must be a slice,a Location or object with a location attribute, not {item}")
        new_record = Record(self.seq[item], f"{self.id}_{item.start}-{item.end}")
        for feature in self.features:  # Assume features are sorted
            if feature.location in item:
                new_record.features.append(new_feature := feature.shift(-item.start))
                new_feature.location.parent_id = new_record.id
        return new_record

    def __repr__(self) -> str:
        return f"{self.id} {self.seq.__repr__()}"

    def __str__(self) -> str:
        return self.id

    def __len__(self) -> int:
        return len(self.seq)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if isinstance(other, Record):
            return self.id == other.id
        return False

    def __iter__(self) -> Iterator['Feature']:
        return iter(self.features)

    def __format__(self, __format_spec: Literal['fasta', 'fna', 'ffn', 'faa', 'bed', 'gfa'] = '') -> str:
        if __format_spec == '':
            return self.__str__()
        elif __format_spec in {'fasta', 'fna'}:
            return f">{self.id}\n{self.seq}\n"
        elif __format_spec in {'faa', 'ffn'}:
            return ''.join(format(i, __format_spec) for i in self.features)
        elif __format_spec == 'bed':
            return ''.join(f"{self.id}\t{format(i, __format_spec)}" for i in self.features)
        elif __format_spec == 'gfa':
            # See: https://gfa-spec.github.io/GFA-spec/GFA1.html
            return f"S\t{self.id}\t{self.desc}\t{self.seq}\n"
        else:
            raise NotImplementedError(f'Invalid format: {__format_spec}')

    def __add__(self, other: 'Record') -> 'Record':
        if isinstance(other, Record):
            new = Record(self.seq + other.seq, f"{self.id}_{other.id}")
            for feature in self.features:
                new_feat = deepcopy(feature)  # Deep copy feature for full independence
                new_feat.location.parent_id = new.id  # Update the ref ID
                new.features.append(new_feat) # Add to new record's features

            for feature in other.features:  # Other feature locations need to be updated
                feature = feature.shift(len(self))  # Creates a new feature and location
                feature.location.parent_id = new.id  # Update the ref ID
                new.features.append(feature)

            new.features.sort(key=attrgetter('location.start'))  # Sort features
            return new
        else:
            raise TypeError(other)

    def __radd__(self, other: 'Record') -> 'Record':
        return other.__add__(self)

    def __iadd__(self, other: 'Record'):
        if isinstance(other, Record):
            self.id = f"{self.id}_{other.id}"  # Update the ref ID
            for feature in self.features:
                feature.location.parent_id = self.id  # Update the ref ID

            for feature in other.features:
                feature = feature.shift(len(self))  # Creates a new feature and location
                feature.location.parent_id = self.id
                self.features.append(feature)  # Update the ref ID

            self.features.sort(key=attrgetter('location.start'))  # Sort features
            self.seq += other.seq  # Now we can update the sequence
            return self
        else:
            raise TypeError(other)

    def __delitem__(self, key: Union[slice, int]):
        """Deletes a slice from the record, adjusting features accordingly."""
        if not isinstance(key, slice):
            raise TypeError(f"Deletion from a Record is only supported for slices, not {type(key)}")

        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("Deletion with a step is not supported.")

        slice_len = stop - start
        if slice_len <= 0:
            return  # Nothing to delete

        new_features = []
        for feature in self.features:
            f_start, f_end = feature.location.start, feature.location.end

            # Case 1: Feature is entirely before the deleted slice
            if f_end <= start:
                new_features.append(feature)  # Feature is before the deleted part
            # Case 2: Feature is entirely after the deleted slice
            elif f_start >= stop:
                new_features.append(feature.shift(-slice_len))
            # Case 3: Feature is partially or fully overlapped
            else:
                # Truncate if it overlaps the start of the slice
                if f_start < start < f_end:
                    feature.location.end = start
                # Truncate and shift if it overlaps the end of the slice
                if f_start < stop < f_end:
                    feature.location.start = stop - slice_len
                # If a feature was modified (i.e., not fully deleted), add it.
                if feature.location.end > feature.location.start:
                    new_features.append(feature)
                # Features fully contained within the slice are implicitly dropped.

        self.features = new_features
        self.seq = self.seq[:start] + self.seq[stop:]

    @classmethod
    def random(cls, id_: str = None, alphabet: Alphabet = DNA, rng: Random = None, gc: float = 0.5,
               length: int = None, min_len: int = 10, max_len: int = 5000000) -> 'Record':
        """
        Generates a random record for testing purposes.

        :param id_: ID of the record. If not provided, a hash will be generated from the sequence.
        :param alphabet: Alphabet of the sequence.
        :param rng: Random number generator.
        :param gc: GC content of the sequence.
        :param length: Length of the sequence. If not provided, a random length will be generated.
        :param min_len: Minimum length of the sequence if length is not specified.
        :param max_len: Maximum length of the sequence if length is not specified.
        :return: A Record instance.
        """
        if alphabet == DNA:
            seq = NucleotideSeq.random(rng, gc, length, min_len, max_len)
        elif alphabet == Protein:
            seq = ProteinSeq.random(rng, length, min_len, max_len)
        else:
            seq = Seq.random(alphabet, rng, length, min_len, max_len)
        return cls(seq, id_)

    def shred(self, rng: Random = None, n_breaks: int = None, break_points: list[int] = None
              ) -> Generator['Record', None, None]:
        """
        Shreds the record into smaller records at the specified break points.

        :param rng: Random number generator
        :param n_breaks: The number of breaks to make in the record. If not provided, a random number of breaks will be
            made between 1 and half the length of the record.
        :param break_points: A list of break points to use. If not provided, random break points will be generated.
        :return: A generator of smaller records
        """
        if rng is None:
            rng = RESOURCES.rng
        if not n_breaks:
            n_breaks = rng.randint(1, len(self) // 2)
        if not break_points:
            break_points = sorted([rng.randint(0, len(self)) for _ in range(n_breaks)])
        previous_end = 0
        for break_point in break_points:
            yield self[previous_end:break_point]
            previous_end = break_point
        yield self[previous_end:]

    def insert(self, other: 'Record', at: int, replace: bool = True) -> 'Record':
        """
        Inserts another record into this record at the specified position.

        :param other: The record to insert.
        :param at: The position to insert the other record at.
        :param replace: Whether to replace the existing sequence at the insertion point with the inserted sequence.
            If False, the inserted sequence will be inserted without removing any existing sequence.
        :return: A new Record instance with the other record inserted.
        """
        if not 0 < at < len(self):
            raise IndexError(f'Cannot insert at {at}, must be between 0 and {len(self)}')
        else:
            new = self[:at] + other + self[at if not replace else at + len(other):]
            return new

    def translate(self, *args, **kwargs) -> 'Record':
        """
        Translates the record to amino acid if it is a nucleotide sequence
        """
        if self.seq.alphabet == DNA:
            return Record(self.seq.translate(*args, **kwargs), self.id)
        return self

    def reverse_complement(self) -> 'Record':
        """
        Returns the reverse complement of the record.
        """
        return Record(self.seq.reverse_complement(), self.id, self.desc, self.qualifiers,
                      [f.reverse_complement(len(self)) for f in self.features[::-1]])

    def add_features(self, *features: HasLocation):
        """
        Adds features to the record.
        :param features: Features to add to the record.
        """
        self.features += features
        self.features.sort(key=attrgetter('location.start'))

    def get_qualifier(self, key: str, default: Any = None):
        return next((v for k, v in self.qualifiers if k == key), default)


class Feature(HasLocation):
    """
    Represents a feature on a sequence, consisting of a location.
    Unlike the BioPython Features, these Features may store their sequences as an attribute, but may not always.

    Attributes:
        id: The unique identifier for the feature.
        kind: The type of feature (e.g., gene, CDS, rRNA).
        seq: The sequence of the feature (optional).
        qualifiers: A list of Qualifier objects providing additional information about the feature.
    """

    def __init__(self, location: Location, kind: str = 'misc_feature', qualifiers: list[Qualifier] = None,
                 seq: Union[Seq, str] = None, id_: str = None):
        super().__init__(location)
        self.kind = kind
        self.qualifiers = deepcopy(qualifiers) if qualifiers is not None else [] # Deep copy qualifiers for full independence
        self.seq = (seq if isinstance(seq, Seq) else Seq(seq)) if seq else None
        self.id = id_ or str(location)

    def __repr__(self):
        return f"{self.kind}({self.id} {self.location})"

    def __str__(self):
        return self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, 'Feature'):
            return self.id == other.id
        return False

    def __getitem__(self, item: str) -> Union[Any, None]:
        """
        Quick method of getting a qualifier by key using next()
        Warning:
            This will only return the first instance, which in most cases is fine
        """
        if not isinstance(item, str):
            raise TypeError(item)
        return next((v for k, v in self.qualifiers if k == item), None)

    def __delitem__(self, item: Union[slice, int, Location, HasLocation]):
        if isinstance(item, Location):
            pass
        elif isinstance(item, HasLocation):
            item = item.location
        elif isinstance(item, slice):
            item = Location.from_slice(item)
        elif isinstance(item, int):
            item = Location.from_slice(slice(item, item))
        else:
            raise TypeError(item)
        del self.location[item]
        self.qualifiers = []
        if self.seq:
            del self.seq[item]

    def __format__(self, __format_spec: Literal['fasta', 'ffn', 'fna', 'faa', 'bed', 'tsv'] = ''):
        if __format_spec == '':
            return self.__str__()
        elif __format_spec in {'fasta', 'ffn', 'fna'}:
            if self.seq is None:
                raise AttributeError(f'No seq extracted for {self=}')
            else:
                return f">{self.id}\n{self.seq}\n"
        elif __format_spec == 'faa':
            return f">{self.id}\n{self.translate()}\n"
        elif __format_spec == 'bed':
            return (f'{self.location.start}\t{self.location.end}\t{self.id}\t'
                    f'{next((v for k, v in self.qualifiers if k == "score"), 0)}\t'
                    f'{decode_strand(self.location.strand)}\n')
        elif __format_spec == 'tsv':
            return f'{self}\t{self.kind}\t{self.location:tsv}'
            # return f'{self}\t{self.location:tsv}'
        else:
            raise NotImplementedError(f'Format "{__format_spec}" not supported')

    @classmethod
    def random(cls, parent: Record, id_: str = None, rng: Random = None, min_len: int = 1,
               max_len: int = 10000, min_start: int = 0, max_start: int = 1000000):
        location = Location.random(rng, min_len=min_len, max_len=min(max_len, len(parent)), min_start=min_start,
                                   max_start=min(max_start, len(parent) - max_len))
        location.parent_id = parent.id
        feature = cls(location, 'CDS')
        feature.extract(parent, store_seq=True)
        return feature

    def translate(
            self,
            store_translation: bool = True,
            parent: Union[Union[Seq, Record, 'Feature'], dict[str, Union[Seq, Record, 'Feature']]] = None,
            store_seq: bool = False, *args, **kwargs
    ) -> Seq:
        """
        Translates the feature to amino acid if it is a nucleotide sequence
        :param store_translation: Boolean to store the translation in the feature
        :param parent: Parent object (``Seq``, ``Record``, ``Feature``) or dictionary of parent objects
        :param store_seq: Boolean to store the translated sequence in the feature
        :return: Translated sequence
        """
        if not (translation := next((v for k, v in self.qualifiers if k == 'translation'), None)):
            if self.seq is None:
                translation = self.extract(parent, store_seq).translate(*args, **kwargs)
            else:
                translation = self.seq.translate(*args, **kwargs)
            if store_translation:
                self.qualifiers.append(Qualifier('translation', translation))
        return translation

    def extract(self, parent: Union[Union[Seq, Record, 'Feature'], dict[str, Union[Seq, Record, 'Feature']]] = None,
                store_seq: bool = False) -> 'Seq':
        seq = self.seq or self.location.extract(parent)
        if store_seq and not self.seq:
            self.seq = seq
        return seq

    def shift(self, by: int) -> 'Feature':
        return Feature(self.location.shift(by), self.kind)

    def reverse_complement(self, parent_length: int) -> 'Feature':
        return Feature(
            self.location.reverse_complement(parent_length), self.kind, qualifiers=deepcopy(self.qualifiers),
            seq=self.seq.reverse_complement() if self.seq else None, id_=self.id
        )

    def complete_CDS(self) -> bool:
        # if not self.kind == 'CDS':
        #     return False
        if not self.seq:
            return False
        if not self.location.partial_start and self.seq[:3] not in DNA.start_codons:
            return False
        if not self.location.partial_end and self.seq[-3:] not in DNA.stop_codons:
            return False
        return True

    def find_promoters(self, parent: Union[Seq, Record], reverse_complement: bool = False, store_seq: bool = False) -> Generator[ 'Feature', None, None]:
        """
        Finds promoters within the feature's sequence.

        The locations of the found promoters are always relative to the parent record's
        coordinate system, accounting for the feature's own position and strand.

        :param parent: The parent Record or Seq object from which the feature was derived. Required for coordinate mapping.
        :param reverse_complement: Whether to search the reverse complement of the feature's sequence.
        :return: A generator of new Feature objects representing the promoters.
        """
        if not isinstance(seq := self.extract(parent, store_seq), NucleotideSeq):
            return  # This check prevents trying to find promoters on a ProteinSeq.

        for n, p_loc in enumerate(seq.find_promoters(reverse_complement), start=1):
            if self.location.strand == 1:
                p_loc = p_loc.shift(self.location.start)
            else:
                p_loc = p_loc.reverse_complement(parent_length=len(self)).shift(self.location.start)

            p_loc.parent_id = self.location.parent_id
            yield Feature(location=p_loc,
                          id_=f"{self.id}_promoter_{n}",
                          kind='regulatory',
                          qualifiers=[Qualifier('regulatory_class', 'promoter')])


# Functions ------------------------------------------------------------------------------------------------------------
def merge_locations(
        locations: Iterable[Union[Location, HasLocation]], tolerance: Union[int, float, None] = None
) -> Generator[Location, None, None]:
    """
    Merges a list of locations into a single location, possibly with a tolerance

    :param locations: Iterable of locations or objects wth a location to merge
    :param tolerance: Tolerance for merging locations or None to merge all locations
    :return: A generator of merged locations

    """
    if not locations:
        return None

    locations = sorted((i.location if isinstance(i, HasLocation) else i for i in locations), key=attrgetter('start'))
    if len(locations) == 1:
        yield from locations
        return None

    if tolerance is None:  # Merge locations regardless
        starts, ends, strands = [], [], []
        for i in locations:
            starts.append(i.start); ends.append(i.end); strands.append(i.strand)
        yield Location(min(starts), max(ends), max(set(strands), key=strands.count))

    else:  # Merge locations within tolerance
        current_location = locations[0]  # Start with first location
        strands = [current_location.strand]  # Collect strands to calculate consensus
        # TODO: Think about how to deal with locations with joins
        # TODO: Also think about how to deal with refs
        for start, end, strand in locations[1:]:  # Iterate through the locations and unpack
            if start - tolerance <= current_location.end:  # Overlap, merge the locations
                current_location = Location(
                    current_location.start, max(current_location.end, end), current_location.strand
                )
                strands.append(strand)
            else:  # No overlap, add the current location to the merged list and start a new location
                current_location.strand = max(set(strands), key=strands.count)
                yield current_location  # Yield the current location
                current_location = Location(start, end, strand)  # Start a new location
                strands = [current_location.strand]

        current_location.strand = max(set(strands), key=strands.count)
        yield current_location  # Yield the last location
