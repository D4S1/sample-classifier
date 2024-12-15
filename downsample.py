import argparse
import random
import pandas as pd
import subprocess
import gzip
from typing import List

import utils
import data.evaluate as evaluate


def downsample_fasta(input_file: str, output_file: str, ratio: float) -> None:
    """
    Downsample a gzipped FASTA file by selecting a random subset of sequences based on the specified ratio.

    :param input_file: Path to the gzipped FASTA input file.
    :param output_file: Path to save the downsampled gzipped FASTA file.
    :param ratio: Ratio of the dataset to retain (0 < ratio <= 1).
    :param city_labels: Dictionary mapping dataset filenames to city labels.
    """
    sequences = []
    headers = []

    with gzip.open(input_file, 'rt') as infile:
        sequence = []
        header = None
        for line in infile:
            line = line.strip()
            if line.startswith(">"):
                if sequence and header:  # Save the previous sequence
                    sequences.append("".join(sequence).upper())
                    headers.append(header)
                header = line
                sequence = []
            else:
                sequence.append(line)
        if sequence and header:  # Save the last sequence
            sequences.append("".join(sequence).upper())
            headers.append(header)

    # Downsample
    total_sequences = len(sequences)
    downsample_size = int(total_sequences * ratio)
    indices = random.sample(range(total_sequences), downsample_size)

    with gzip.open(output_file, 'wt') as outfile:
        for i in indices:
            outfile.write(f"{headers[i]}\n")
            outfile.write(f"{sequences[i]}\n")


def evaluate_downsampling(training_file: str, testing_file: str, ground_truth_file: str, output_prefix: str, ratios: List[float]):
    """
    Evaluate the classifier performance on downsampled gzipped FASTA files.

    :param training_file: Path to the original training gzipped FASTA file.
    :param testing_file: Path to the original testing gzipped FASTA file.
    :param ground_truth_file: Path to the ground truth file for evaluation.
    :param output_prefix: Prefix for saving outputs and downsampled files.
    :param ratios: List of ratios (e.g., [0.1, 0.2, ..., 1.0]) to downsample the data.
    :param city_labels: Dictionary mapping dataset filenames to city labels.
    """

    city_labels = list(set(utils.load_ref(training_file).values()))
    results = []


    for ratio in ratios:
        print(f"Evaluating with ratio: {ratio}")

        # Downsample training data
        train_downsampled = f"{output_prefix}_train_{int(ratio * 100)}.fasta.gz"
        downsample_fasta(training_file, train_downsampled, ratio, city_labels)

        # Downsample testing data
        test_downsampled = f"{output_prefix}_test_{int(ratio * 100)}.fasta.gz"
        downsample_fasta(testing_file, test_downsampled, ratio, city_labels)

        # Run the classifier
        classifier_output = f"{output_prefix}_output_{int(ratio * 100)}.tsv"
        subprocess.run([
            "python", "classifier.py",
            train_downsampled, test_downsampled, classifier_output
        ])

        # Evaluate the output
        auc_score = evaluate.calculate_auc_roc(classifier_output, ground_truth_file)
        results.append({
            'ratio': ratio,
            'auc': auc_score
        })

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_prefix}_results.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier performance on downsampled gzipped FASTA data.")
    parser.add_argument("training_data", type=str, help="Path to the original training gzipped FASTA file.")
    parser.add_argument("testing_data", type=str, help="Path to the original testing gzipped FASTA file.")
    parser.add_argument("ground_truth", type=str, help="Path to the ground truth file for evaluation.")
    parser.add_argument("output_prefix", type=str, help="Prefix for saving outputs and downsampled files.")
    parser.add_argument("--ratios", type=float, nargs='+', default=[0.1, 0.2, 0.5, 0.75, 1.0], help="List of downsampling ratios.")
    args = parser.parse_args()


    evaluate_downsampling(
        training_file=args.training_data,
        testing_file=args.testing_data,
        ground_truth_file=args.ground_truth,
        output_prefix=args.output_prefix,
        ratios=args.ratios
    )

if __name__ == "__main__":
    main()
