import utils
import mmh3
import numpy as np
from typing import List, Set, Dict

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
    Preprocesses a dataset by extracting k-mers and returning a sketch of the dataset.

    :param dataset: List of sequneces
    :param k: k-mer size.
    :param s: Sketch size.
    :param seed: Random seed for reproducibility.
    :param ci: minimum  kmer frequency
    :return: Sketch of the k-mer set with size s.
    """
    kmers = set()
    for read in dataset:
        kmers = kmers.union(sketch(kmer_set(read, k, seed, ci), s))
    return sketch(kmers, s)


def preprocess_reference(train_filename: str, k: int, s: int, human_set: set, seed: int, ci: int) -> dict:
    """
    Preprocesses a dataset by extracting k-mers and returning a dictionary with skeuches of the cities.

    :param train_filename: file containing meta filenames of reference datasets with their classification.
    :param k: kmer size.
    :param s: sketch size.
    :param seed: Random seed for reproducibility.
    :param ci: minimum  kmer frequency
    :return: dictionary of the k-mer sketches of size s representing cities.
    """
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


def estimate_jackard(read_sketch: set, city_sketch: set, s: int) -> float:
    """
    Estimates the Jaccard similarity between a read sketch and a city sketch.

    :param read_sketch: Sketch of the read.
    :param city_sketch: Sketch of the city.
    :param s: Sketch size.
    :return: Estimated Jaccard similarity.
    """
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


def calculate_scores(read_scores: np.array, threshold: float, max_matches: int) -> dict:
    """
    Calculate simple, fractional, and weighted scores for each class based on match rate scores.

    :param read_scores: NumPy array of shape (number of reads, number of classes) containing match rate scores for reads.
    :param threshold: Minimum score to consider a match.
    :param max_matches: Maximum number of matches allowed per read.
    :return: Dictionary containing simple, fractional, and weighted scores for each class.
    """
    n_col = read_scores.shape[1]
    scores = {
        "simple": [0] * n_col,
        "fractional": [0] * n_col,
        "weighted": [0] * n_col,
    }

    for read in read_scores:
        matching_classes = [(city, score) for city, score in enumerate(read) if score >= threshold]
        if len(matching_classes) <= max_matches:

            n = len(matching_classes)
            total_score = sum(score for _, score in matching_classes)

            for city, score in matching_classes:
                scores['simple'][city] += 1
                if n > 0:
                    scores['fractional'][city] += 1 / n
                if total_score > 0:
                    scores['weighted'][city] += score / total_score
    return scores

def preprocess_sample(filename: str, human_sketch: set, k: int, s: int, seed: int, ci: int) -> List[set]:
    """
    Preprocess a sample by generating sketches for each read after filtering human sequences.

    :param filename: Path to the input FASTA file.
    :param k: Length of k-mers to generate.
    :param s: Size of the sketch.
    :param seed: Seed for hash functions used in sketching.
    :param ci: Context-specific parameter for generating k-mers.
    :return: List of sketches (sets) representing the reads in the sample.
    """
    dataset = utils.load_dataset(filename)
    kmers_sets = []

    for read in dataset:
        kmers = kmer_set(read, k, seed, ci)
        sketch = sketch(kmers, s)
        if len(sketch & human_sketch) == 0:  # Filter out reads overlapping with human sequences
            kmers_sets.append(sketch)
            
    return kmers_sets

def classify_sample(sample_sketches: List[Set[int]], reference: Dict[str, Set[int]]) -> np.array:
    """
    Classify a sample by comparing read sketches against reference sketches.

    :param sample_sketches: List of sketches representing the reads in the sample.
    :param reference: Dictionary of reference sketches (key: class name, value: sketch set).
    :return: NumPy array of shape (number of reads, number of reference classes) with similarity scores.
    """
    n_reads = len(sample_sketches)
    n_classes = len(reference)
    score_matrix = np.zeros((n_reads, n_classes))

    reference_items = list(reference.items())
    for i, read_sketch in enumerate(sample_sketches):
        for j, (class_name, class_sketch) in enumerate(reference_items):
            score_matrix[i, j] = estimate_jaccard(read_sketch, class_sketch, len(read_sketch))

    return score_matrix

def classify_samples(samples_filenames: List[str], cities_labels: List[str], k: int, s: int, seed: int, ci: int) -> Dict[str, np.array]:
    """
    Classify multiple samples, calculating scores for each sample and reference class.

    :param samples_filenames: List of file paths for input samples.
    :param cities_labels: List of reference class names.
    :param k: Length of k-mers to generate.
    :param s: Size of the sketch.
    :param seed: Seed for hash functions used in sketching.
    :param ci: Context-specific parameter for generating k-mers.
    :return: Dictionary containing classification matrices for each score type.
    """
    n_samples = len(samples_filenames)
    n_classes = len(cities_labels)

    results = {
        "simple": np.zeros((n_samples, n_classes)),
        "fractional": np.zeros((n_samples, n_classes)),
        "weighted": np.zeros((n_samples, n_classes)),
    }

    for sample_idx, filename in enumerate(samples_filenames):
        sample_sketches = preprocess_sample(filename, k, s, seed, ci)
        score_matrix = classify_sample(sample_sketches, {label: get_city_sketch(label) for label in cities_labels})
        sample_scores = calculate_scores(score_matrix, threshold=0.5, max_matches=5)  # Example threshold/max_matches

        for score_type in results.keys():
            results[score_type][sample_idx, :] = sample_scores[score_type]

    return results

