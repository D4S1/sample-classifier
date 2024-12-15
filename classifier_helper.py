import utils
import mmh3
import numpy as np
from typing import List, Set, Dict
import gzip
import re
import os


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


def preprocess_dataset(filename: str, k: int, seed: int, ci: int, dir: str="data/"):
    """
    Preprocesses a dataset by extracting k-mers and returning a sketch of the dataset.

    :param dataset: List of sequneces
    :param k: k-mer size.
    :param seed: Random seed for reproducibility.
    :param ci: minimum  kmer frequency
    :return: Sketch of the k-mer set with size s.
    """
    ci = 1
    dataset_sketch = set()
    print(filename)
    with gzip.open(f'{dir}{filename}', 'rt') as f:
        while True:
            chunk = f.read(10**6)
            chunk = re.sub(r'>.*\n', 'N', chunk)
            chunk = chunk.replace('\n', '').upper()

            chunk_kmers = kmer_set(chunk, k, seed, ci)
            dataset_sketch = dataset_sketch.union(chunk_kmers)
            if len(dataset_sketch) >= 10**6:
                dataset_sketch = set(list(dataset_sketch)[:10**5])
            if not chunk:
                break
    return set(list(dataset_sketch)[:10**4])

def preprocess_human(filename: str, k: int, seed: int, ci: int):
    """
    Preprocesses a dataset by extracting k-mers and returning a sketch of the dataset.

    :param dataset: List of sequneces
    :param k: k-mer size.
    :param seed: Random seed for reproducibility.
    :param ci: minimum  kmer frequency
    :return: Sketch of the k-mer set with size s.
    """
    ci = 1
    dataset_sketch = set()
    with open(filename, 'r') as f:
        while True:
            chunk = f.read(10**6)
            chunk = re.sub(r'>.*\n', 'N', chunk)
            chunk = chunk.replace('\n', '').upper()

            chunk_kmers = kmer_set(chunk, k, seed, ci)
            dataset_sketch = dataset_sketch.union(chunk_kmers)
            print(f'{len(dataset_sketch)=}')
            if len(dataset_sketch) >= 10**6:
                dataset_sketch = set(list(dataset_sketch)[:10**5])
            if not chunk:
                break
    return set(list(dataset_sketch)[:10**4])


def preprocess_reference(train_filename: str, k: int, human_set: set, seed: int, ci: int) -> dict:
    """
    Preprocesses a dataset by extracting k-mers and returning a dictionary with skeuches of the cities.

    :param train_filename: file containing meta filenames of reference datasets with their classification.
    :param k: kmer size.
    :param seed: Random seed for reproducibility.
    :param ci: minimum  kmer frequency
    :return: dictionary of the k-mer sketches of size s representing cities.
    """
    train_data = utils.load_ref(train_filename)
    city_labels = list(set(train_data.values()))
    cities_sketches = {city : set() for city in city_labels}  # {city : set of dataset sketches}

    for filename, city in train_data.items():
        dataset_sketch = preprocess_dataset(filename, k=k, seed=seed, ci=ci, dir=os.path.dirname(train_filename)+'/')
        cities_sketches[city] = cities_sketches[city].union(dataset_sketch)
    
    for city, sketch_set in cities_sketches.items():
        cities_sketches[city] = filter_human(sketch_set, human_set)

    return city_labels, cities_sketches


def jackard_score(read_kmers: list, city_kmers: set) -> float:
    """
    Estimates the Jaccard similarity between a read sketch and a city sketch.

    :param read_sketch: Sketch of the read.
    :param city_sketch: Sketch of the city.
    :return: Estimated Jaccard similarity.
    """
    return len(set(read_kmers) & city_kmers) / len(set(read_kmers) | city_kmers)

def cometa_score(read_kmers: list, city_kmers: set, k: int) -> float:
    """
    Computes the CoMeta score for a given list of kmers based on matches with a reference set of kmers.

    :param read_kmers: List of kmers from the read to be scored.
    :param city_kmers: Set of kmers from the reference city to match against.
    :param k: Length of the kmers used for scoring.
    :return: Normalized CoMeta score as a float.
    """
    last = -k
    score = 0
    for i, kmer in enumerate(read_kmers):
        if kmer in city_kmers:
            score += k - max(k + last - i, 0)
            last = i
    return score / (len(read_kmers) + 2* k - 2)


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
        if read.sum() == 0:
            continue
        
        matching_classes = [(city, score) for city, score in enumerate(read) if score >= threshold]

        if len(matching_classes) == 0:
            max_value = np.max(read)
            matching_classes= [(pos, max_value) for pos in np.where(read == max_value)[0]]

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

def preprocess_sample(filename: str, human_sketch: set, k: int, seed: int, dir: str='data/') -> List[set]:
    """
    Preprocess a sample by generating sketches for each read after filtering human sequences.

    :param filename: Path to the input FASTA file.
    :param k: Length of k-mers to generate.
    :param s: Size of the sketch.
    :param seed: Seed for hash functions used in sketching.
    :param ci: Context-specific parameter for generating k-mers.
    :return: List of sketches (sets) representing the reads in the sample.
    """
    dataset = utils.load_dataset(filename, dir=dir)
    reads_kmers = []

    for read in dataset:
        read = read + "N" + reverse_complement(read)
        read_kmers = [mmh3.hash(read[i:i+k], seed) for i in range(len(read) - k + 1) if 'N' not in read[i:i+k]]
        if len(set(read_kmers) & human_sketch) == 0:  # Filter out reads overlapping with human sequences
            reads_kmers.append(read_kmers)
    return reads_kmers

def classify_sample(sample_sketches: List[set], reference: Dict[str, set], city_labes: List[str], k: int) -> np.array:
    """
    Classify a sample by comparing read sketches against reference sketches.

    :param sample_sketches: List of sketches representing the reads in the sample.
    :param reference: Dictionary of reference sketches (key: class name, value: sketch set).
    :return: NumPy array of shape (number of reads, number of reference classes) with similarity scores.
    """
    n_reads = len(sample_sketches)
    n_cities = len(city_labes)
    score_matrix = np.zeros((n_reads, n_cities))

    for i, read_kmers in enumerate(sample_sketches):
        for j, city in enumerate(city_labes):
            score_matrix[i, j] = cometa_score(read_kmers, reference[city], k)

    return score_matrix

def classify_samples(test_data_file: str, output_file: str, reference_data: dict, city_labels: List[str], human_set: set, k: int, seed: int, M: int, T: int) -> Dict[str, np.array]:
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
    n_samples = len(samples_filenames)
    n_classes = len(city_labels)
    results = {
        "simple": np.zeros((n_samples, n_classes)),
        "fractional": np.zeros((n_samples, n_classes)),
        "weighted": np.zeros((n_samples, n_classes)),
    }

    for sample_idx, filename in enumerate(samples_filenames):
        print(f'{sample_idx=}')
        sample_sketches = preprocess_sample(filename, human_set, k, seed, dir=os.path.dirname(test_data_file)+'/') # list of kmers lists
        score_matrix = classify_sample(sample_sketches, reference_data, city_labels, k)
        sample_scores = calculate_scores(score_matrix, T, max_matches=M)  # Example threshold/max_matches
        for score_type in results.keys():
            results[score_type][sample_idx, :] = sample_scores[score_type]

    utils.save_to_file(samples_filenames, city_labels, results[score_type], output_file)

    return score_matrix

def reverse_complement(seq: str) -> str:
    """
    Computes the reverse complement of a DNA sequence.

    :param seq: DNA sequence as a string, which may include both uppercase and lowercase bases.
    :return: Reverse complement of the input DNA sequence as a string.
    """
    complement = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
        'N': 'N',
        'a': 't',
        't': 'a',
        'c': 'g',
        'g': 'c',
        'n': 'N'
    }
    
    rev_seq = ['A'] * len(seq)  # arbitrary initialization of reversed complement sequence
    for i in range(len(seq)-1, -1, -1): 
        rev_seq[len(seq)-1-i] = complement[seq[i]]
    return ''.join(rev_seq)
