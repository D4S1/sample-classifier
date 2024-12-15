import classifier_helper
import utils
import time
import argparse

def main(train_data_file: str, test_data_file: str, output_file: str, k: int=24, ci: int = 4, seed=12345, M: int = 3, threshold: float = 0.265):

    human_sketch = utils.load_pickle('human_sketch.pkl')
    # start = time.time()
    # human_sketch = classifier_helper.preprocess_human('gencode.v47.transcripts.fa', k, seed, ci)
    # print(f'human sketch prep: {(time.time()-start)/60} min')
    # print(f'human sketch len {len(human_sketch)}')
    print("Prepering the reference")
    start = time.time()
    city_labels, cities_sketches = classifier_helper.preprocess_reference(train_data_file, k=k, human_set = human_sketch, seed=seed, ci=ci)
    print(f'reference prep: {(time.time()-start)/60} min')

    print("Classification")
    start = time.time()
    sample_classification = classifier_helper.classify_samples(test_data_file, output_file, cities_sketches, city_labels, human_sketch, k, seed, M=M, T=threshold)
    print(f'classification: {(time.time()-start)/60} min')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify environmental metagenomics samples to provided reference data")
    parser.add_argument("training_data", type=str, help="File which containts datasets filename and names of the city from which the samples was derived")
    parser.add_argument("testing_data", type=str, help="File which containts test dataset filename")
    parser.add_argument("output", type=str, help="Path were the results will be saved to")
    args = parser.parse_args()
    main(args.training_data, args.testing_data, args.output)
