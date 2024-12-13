import utils
import mmh3
import numpy as np
from typing import List, Set, Dict
import gzip
import re


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
            kmer_counts[hashed_kmer] = kmer_counts.get(hashed_kmer, 0) + 1
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


def preprocess_dataset(filename: str, k: int, s: int, seed: int, ci: int):
    """
    Preprocesses a dataset by extracting k-mers and returning a sketch of the dataset.

    :param dataset: List of sequneces
    :param k: k-mer size.
    :param s: Sketch size.
    :param seed: Random seed for reproducibility.
    :param ci: minimum  kmer frequency
    :return: Sketch of the k-mer set with size s.
    """
    ci = 1
    dataset_sketch = set()
    with gzip.open(f'data/{filename}', 'rt') as f:
        while True:
            chunk = f.read(10**6)
            chunk = re.sub(r'>.*\n', 'N', chunk)
            chunk = chunk.replace('\n', '').upper()

            chunk_kmers = kmer_set(chunk, k, seed, ci)
            dataset_sketch = dataset_sketch.union(chunk_kmers)
            if len(dataset_sketch) >= 50**6:
                dataset_sketch = sketch(dataset_sketch, s)
            if not chunk:
                break
    return sketch(dataset_sketch, s)


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
        dataset_sketch = preprocess_dataset(filename, k=k, s=s, seed=seed, ci=ci)
        cities_sketches[city] = cities_sketches[city].union(dataset_sketch)
    
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
        print(f'{len(matching_classes)}')
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

def preprocess_sample(filename: str, human_sketch: set, k: int, s: int, seed: int, ci: int, bin_size: int) -> List[set]:
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
    reads_sketches = []
    bin, bin_counter = set(), 0
    for read in dataset:
        read_kmers = kmer_set(read, k, seed, ci=1)
        if len(read_kmers & human_sketch) == 0:  # Filter out reads overlapping with human sequences
            bin = bin.union(read_kmers)
            bin_counter += 1
        if bin_size <= bin_counter:
            reads_sketches.append(sketch(bin, s))
            bin, bin_counter = set(), 0
    if bin_size <= bin_counter:
        reads_sketches.append(sketch(bin, s))
    return reads_sketches

def classify_sample(sample_sketches: List[set], reference: Dict[str, Set[int]]) -> np.array:
    """
    Classify a sample by comparing read sketches against reference sketches.

    :param sample_sketches: List of sketches representing the reads in the sample.
    :param reference: Dictionary of reference sketches (key: class name, value: sketch set).
    :return: NumPy array of shape (number of reads, number of reference classes) with similarity scores.
    """
    n_reads = len(sample_sketches)
    n_cities = len(reference)
    score_matrix = np.zeros((n_reads, n_cities))

    for i, read_sketch in enumerate(sample_sketches):
        for j, (class_name, class_sketch) in enumerate(reference.items()):
            score_matrix[i, j] = estimate_jackard(read_sketch, class_sketch, len(read_sketch))

    return score_matrix

def classify_samples(test_data_file: str, output_file: str, reference_data: dict, human_set: set, k: int, s: int, seed: int, ci: int, threshold: float, bin_size: int) -> Dict[str, np.array]:
    """
    Classify multiple samples, calculating scores for each sample and reference class.

    :param samples_filenames: List of file paths for input samples.
    :param cities_labels: List of reference class names.
    :param k: Length of k-mers to generate.
    :param s: Size of the sketch.
    :param seed: Seed for hash functions used in sketching.
    :param ci: Context-specific parameter for generating k-mers.
    :param bin_size: Size of bin utilse to create comparable size skeches for sample
    :return: Dictionary containing classification matrices for each score type.
    """
    samples_filenames = utils.load_samples(test_data_file)
    city_labels = list(reference_data.keys())
    n_samples = len(samples_filenames)
    n_classes = len(reference_data)
    results = {
        "simple": np.zeros((n_samples, n_classes)),
        "fractional": np.zeros((n_samples, n_classes)),
        "weighted": np.zeros((n_samples, n_classes)),
        'jc': np.zeros((n_samples, n_classes))
    }

    for sample_idx, filename in enumerate(samples_filenames):
        print(f'{sample_idx=}')
        sample_sketches = preprocess_sample(filename, human_set, k, s, seed, ci, bin_size)
        score_matrix = classify_sample(sample_sketches, reference_data)
        results['jc'][sample_idx, :] = np.sum(score_matrix, axis=0)
        # sample_scores = calculate_scores(score_matrix, threshold, max_matches=5)  # Example threshold/max_matches
        # for score_type in results.keys():
        #     results[score_type][sample_idx, :] = sample_scores[score_type]

    # for score_type in results.keys():
    #    utils.save_to_file(samples_filenames, city_labels, results[score_type], f'data/outs/{output_file}_{score_type}.tsv')
    utils.save_to_file(samples_filenames, city_labels, results['jc'], f'data/outs/{output_file}_jc.tsv')
    return score_matrix

