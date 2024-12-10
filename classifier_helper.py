import utils
import mmh3
from collections import Counter

# ci = 4
# T = 50% ?
# M = 8 ?
# k = 24
# s = 1000?


def kmer_set(seq: str, k: int, seed: int, ci: int) -> set:
    """
    Generate a set of unique k-mers from a sequence, filtered by occurrence count.

    :param seq: Read sequence (string).
    :param k: Length of k-mer (int).
    :param seed: Seed for hash function (int).
    :param ci: Minimum count threshold for k-mers (int).
    :return: Set of unique k-mers appearing at least 'ci' times.
    """
    kmer_counts = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if not 'N' in kmer:
            hashed_kmer = mmh3.hash(kmer, seed)
            kmer_counts[kmer] = kmer_counts.get(hashed_kmer, 0) + 1

    return {kmer for kmer, count in kmer_counts.items() if count >= ci}


def sketch(kmer_set: str, s: int) -> set:
    """
    Takes a set of kmers and return set of s smallest ones

    :param kmer_set: set of hashed or not kmers
    :param s: size of sketch
    :returns: set of s smallest kmers
    """
    return sorted(kmer_set)[:s]


def filter_human(input_sketch: set, human_set: set) -> set:
    """
    Filters out elements in the input sketch that are present in the human set.
    
    :param input_sketch: Set of hashed k-mers from the input dataset.
    :param human_set: Set of hashed k-mers representing human sequences.
    :return: Set of k-mers from input_sketch not present in human_set.
    """
    return {kmer for kmer in input_sketch if kmer not in human_set}


def preprocess_dataset(dataset: ..., k: int, s: int) -> set:
    """
    For each dataset, loop over reads and add k-mers to the set, return sketch of the set 
    """

def preprocess_reference(train_filename: str, k: int, s: int, human_set: set):
    train_data = utils.load_ref(train_filename)
    cities_sketches = {city : set() for city in set(train_data.values())}  # {city : set of dataset sketches}

    for filename, city in train_data.items():
        dataset = utils.load_dataset(filename)
        cities_sketches[city] = cities_sketches[city].union(preprocess_dataset(dataset, k, s))
    
    for city, sketch_set in cities_sketches.items():
        sketch_set = filter_human(sketch_set, human_set)
        try:
            cities_sketches[city] = sketch(sketch_set, s)
        except Exception as e:
            print(f'Sketch for {city} has less than s = {s} elements\n{e}')

    return cities_sketches

def preprocess_human_genome(filename: str) -> str:
    pass

def estimate_jackard(read_sketch: str, city_sketch: set, s: int) -> float:
    return len(sketch(read_sketch.union(city_sketch), s).intersection(read_sketch).intersection(city_sketch)) / s
