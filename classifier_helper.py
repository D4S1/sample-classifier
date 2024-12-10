import utils
import mmh3

# ci = 4
# T = 50% ?
# M = 8 ?
# k = 24
# s

def kmer_set(seq: str, k: int, seed: int) -> set:
    """
    :param seq: read sequence
    :param k: lenght of kmer
    :return: set of unique kmers in given sequence
    """
    unique_kmers = set()
    for i in range(len(seq) - k + 1):
        unique_kmers.add(mmh3.hash(seq[i:i+k], seed))
    return unique_kmers


def sketch(kmer_set: str, s: int) -> set:
    """
    Takes a set of kmers and return set of s smallest ones

    :param kmer_set: set of hashed or not kmers
    :param s: size of sketch
    :returns: set of s smallest kmers
    """
    return sorted(kmer_set)[:s]

def filter_human(input_sketch: set, human_set: set) -> set:
    pass

def preprocess_dataset(dataset: ..., k: int, s: int) -> set:
    """
    For each dataset, loop over reads and add k-mers to the set, return sketch of the set 
    """

def preprocess_reference(train_filename: str, k: int, s: int):
    train_data = utils.load_ref(train_filename)
    cities_sketches = {city : set() for city in set(train_data.values())}  # {city : set of dataset sketches}

    for filename, city in train_data.items():
        dataset = utils.load_dataset(filename)
        cities_sketches[city] = cities_sketches[city].union(preprocess_dataset(dataset, k, s))
    
    for city, sketch_set in cities_sketches.items():
        cities_sketches[city] = sketch(sketch_set, s)

    return cities_sketches
    




