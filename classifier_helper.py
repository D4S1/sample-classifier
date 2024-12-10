import utils
import mmh3
import numpy as np

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
    print(kmer_counts)
    return {kmer for kmer, count in kmer_counts.items() if count >= ci}


def sketch(kmer_set: str, s: int) -> set:
    """
    Takes a set of kmers and return set of s smallest ones

    :param kmer_set: set of hashed or not kmers
    :param s: size of sketch
    :returns: set of s smallest kmers
    """
    return set(sorted(kmer_set)[:s])


def filter_human(input_sketch: set, human_set: set) -> set:
    """
    Filters out elements in the input sketch that are present in the human set.
    
    :param input_sketch: Set of hashed k-mers from the input dataset.
    :param human_set: Set of hashed k-mers representing human sequences.
    :return: Set of k-mers from input_sketch not present in human_set.
    """
    return input_sketch.difference(human_set)


def human_sketch(filename: str, k: int, s: int, seed: int, ci: int):
    genome = utils.load_dataset2(filename)
    genome = preprocess_dataset(genome, k, s, seed, ci)
    return genome


def preprocess_dataset(dataset: list[str], k: int, s: int, seed:int, ci:int) -> set:
    """
    For given dataset, loop over reads and add k-mers to the result set, return sketch of the set 
    """
    kmers = set()
    for read in dataset:
        kmers = kmers.union(sketch(kmer_set(read, k, seed, ci), s))
    return sketch(kmers, s)


def preprocess_reference(train_filename: str, k: int, s: int, human_set: set, seed: int, ci: int):
    train_data = utils.load_ref(train_filename)
    cities_sketches = {city : set() for city in set(train_data.values())}  # {city : set of dataset sketches}

    for filename, city in train_data.items():
        dataset = utils.load_dataset(filename)
        cities_sketches[city] = cities_sketches[city].union(preprocess_dataset(dataset, k, s, seed, ci))
    
    for city, sketch_set in cities_sketches.items():
        sketch_set = filter_human(sketch_set, human_set)
        try:
            cities_sketches[city] = sketch(sketch_set, s)
        except Exception as e:
            print(f'Sketch for {city} has less than s = {s} elements\n{e}')

    return cities_sketches


def estimate_jackard(read_sketch: str, city_sketch: set, s: int) -> float:
    return len(sketch(read_sketch.union(city_sketch), s).intersection(read_sketch).intersection(city_sketch)) / s


def simple_sum(jackard_estimates: np.ndarray, T: float) -> np.ndarray:
    """
    Matches sample to city via simple sum criterion.

    :param jackard_estimates: NumPy array of shape (number of reads, number of cities).
    :param T: Threshold value for comparison.
    :return: Updated NumPy array with values as 0 or 1.
    """
    jackard_estimates = (jackard_estimates >= T).astype(int)
    cities_sums = jackard_estimates.sum(axis=0)
    return np.argmax(cities_sums)  # return maximum column index

human_sketch = human_sketch('GCA_000001405.15_GRCh38_genomic.fna', k=24, s=1000, seed=12345, ci=4)
print(f'{type(human_sketch)=}\t {len(human_sketch)=}')
utils.save_to_file(human_sketch, 'human_sketch.pkl')