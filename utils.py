from typing import List
import numpy as np
import gzip
import pickle


def load_ref(filename: str) -> dict:
    """
    Load training data, which has following columns of interest for each dataset (row):
    fasta_file, geo_loc_name
    :param filename: path to the training data
    :return:
    """
    ref_data = {}
    with open(filename) as f:
        next(f)
        for line in f:
            dataset, city, *_ = line.split("\t")
            ref_data[dataset] = city
    
    return ref_data

def load_samples(filename: str) -> list:
    """
    Load test data filenames
    :param filename: path to the test data
    :return:
    """
    with open(filename) as f:
        samples = f.read().strip().split("\n")[1:]
    return samples

def load_dataset(filename: str) -> List[str]:
    """
    Load sequences from a gzipped FASTA file.
    :param filename (str): Path to the gzipped FASTA file
    :return: list of sequences as strings
    """
    sequences = []
    with gzip.open(filename, 'rt') as file:  # open in text mode
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:  # if there's an ongoing sequence, save it
                    sequences.append("".join(sequence).upper())
                    sequence = []  # reset for the next sequence
            else:
                sequence.append(line)  # add to the current sequence
        if sequence:  # append the last sequence
            sequences.append("".join(sequence).upper())
    return sequences

def load_dataset2(filename: str) -> List[str]:
    """
    Load sequences from a gzipped FASTA file.
    :param filename (str): Path to the gzipped FASTA file
    :return: list of sequences as strings
    """
    sequences = []
    with open(filename, 'r') as file:  # open in text mode
        sequence = []
        i = 0
        for line in file:
            if i == 3:
                break
            line = line.strip()
            if line.startswith(">"):
                if sequence:  # if there's an ongoing sequence, save it
                    sequences.append("".join(sequence).upper())
                    sequence = []  # reset for the next sequence
                    i += 1
            else:
                sequence.append(line)  # add to the current sequence
        if sequence:  # append the last sequence
            sequences.append("".join(sequence).upper())
    return sequences


def save_to_file(dataset_names: List[str], city_labels: List[str], classification_matrix: np.array, output_file: str) -> None:
    """
    Saves results to output format provided in task distription
    header line with columns names: fasta_file and city names
    body: dataset filenames, likeliness of being classied to city
    :param dataset_names: List of dataset filenames
    :param city labels: list of unique city names from reference data
    :param classification_matrix: matrix with shape (number of dataset, number of unique cities)
    containg likelihood of sample beeing colected from city from column i
    :param output_file: output filename where the results will be saved
    :returns: None
    """

    with open(output_file, 'w') as f:
        header = "fasta_file\t" + "\t".join(city_labels) + "\n"
        f.write(header)

        for i, dataset in enumerate(dataset_names):
            f.write(f"{dataset}\t")
            for j, city_prob in enumerate(classification_matrix[i]):
                if j == len(classification_matrix[i]) - 1:
                    f.write(f"{city_prob}\n")
                    break
                f.write(f"{city_prob}\t")
                
def save_to_file(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
