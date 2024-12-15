import argparse
import gzip
import random
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score


def parse_training_data(filename):
    """
    Parse the training data metadata.
    :param filename: Path to the training data metadata file.
    :return: Dictionary mapping FASTA filenames to their class (city labels).
    """
    training_data = {}
    with open(filename, 'r') as file:
        next(file)  # Skip header
        for line in file:
            columns = line.strip().split('\t')
            fasta_file, city = columns[:2]  # First two columns are relevant
            training_data[fasta_file] = city
    return training_data


def parse_testing_data(filename):
    """
    Parse the testing data metadata.
    :param filename: Path to the testing data metadata file.
    :return: List of FASTA filenames.
    """
    with open(filename, 'r') as file:
        return [line.strip().split('\t')[0] for line in file.readlines()[1:]]


def load_fasta_sequences(filename):
    """
    Load sequences from a gzipped FASTA file.
    :param filename: Path to the gzipped FASTA file.
    :return: List of sequences.
    """
    sequences = []
    with gzip.open(filename, 'rt') as file:
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(''.join(sequence))
                    sequence = []
            else:
                sequence.append(line)
        if sequence:
            sequences.append(''.join(sequence))
    return sequences


def downsample_sequences(sequences, ratio):
    """
    Downsample a list of sequences based on the given ratio.
    :param sequences: List of sequences.
    :param ratio: Ratio of sequences to retain (0 < ratio <= 1).
    :return: Downsampled list of sequences.
    """
    num_to_sample = int(len(sequences) * ratio)
    return random.sample(sequences, num_to_sample)


def generate_class_likelihoods(training_data, testing_data, output_file):
    """
    Generate class likelihoods for testing datasets based on simple sequence counts.
    :param training_data: Dictionary mapping training FASTA filenames to class labels.
    :param testing_data: List of testing FASTA filenames.
    :param output_file: Path to the output file.
    """
    # Count sequences for each class in the training data
    class_counts = Counter()
    for fasta_file, class_label in training_data.items():
        sequences = load_fasta_sequences(f"data/{fasta_file}")
        class_counts[class_label] += len(sequences)

    # Prepare output
    class_labels = sorted(class_counts.keys())
    results = []

    for fasta_file in testing_data:
        sequences = load_fasta_sequences(f"data/{fasta_file}")
        seq_count = len(sequences)

        likelihoods = {label: seq_count / class_counts[label] if class_counts[label] > 0 else 0
                       for label in class_labels}

        results.append([fasta_file] + [likelihoods[label] for label in class_labels])

    # Write to output file
    with open(output_file, 'w') as file:
        file.write('\t'.join(['fasta_file'] + class_labels) + '\n')
        for result in results:
            file.write('\t'.join(map(str, result)) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Generate class likelihoods for testing datasets.")
    parser.add_argument("training_metadata", type=str, help="Path to the training metadata file.")
    parser.add_argument("testing_metadata", type=str, help="Path to the testing metadata file.")
    parser.add_argument("output_file", type=str, help="Path to the output file.")
    parser.add_argument("--downsample_ratio", type=float, default=1.0, help="Ratio to downsample training data.")

    args = parser.parse_args()

    # Parse input metadata
    training_data = parse_training_data(args.training_metadata)
    testing_data = parse_testing_data(args.testing_metadata)

    # Optionally downsample training data
    if args.downsample_ratio < 1.0:
        for fasta_file in training_data:
            sequences = load_fasta_sequences(f"data/{fasta_file}")
            downsampled = downsample_sequences(sequences, args.downsample_ratio)

            # Save downsampled data back to file (overwrite for simplicity)
            with gzip.open(f"data/{fasta_file}", 'wt') as file:
                for i, sequence in enumerate(downsampled):
                    file.write(f">seq{i}\n{sequence}\n")

    # Generate class likelihoods
    generate_class_likelihoods(training_data, testing_data, args.output_file)


if __name__ == "__main__":
    main()
