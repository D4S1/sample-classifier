from typing import List
import numpy as np


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

