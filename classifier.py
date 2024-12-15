import classifier_helper
import utils
import time

def main(train_data_file: str, test_data_file: str, output_file: str, k: int=24, ci: int = 4, seed=12345, M: int = 3, threshold: float = 0.265):

    human_sketch = utils.load_pickle('human_sketch.pkl')
    # start = time.time()
    # human_sketch = classifier_helper.preprocess_human('gencode.v47.transcripts.fa', k, seed, ci)
    # print(f'human sketch prep: {(time.time()-start)/60} min')
    print(f'human sketch len {len(human_sketch)}')
    
    start = time.time()
    city_labels, cities_sketches = classifier_helper.preprocess_reference(train_data_file, k=k, human_set = human_sketch, seed=seed, ci=ci)
    print(f'reference prep: {(time.time()-start)/60} min')
    for city, sketch in cities_sketches.items():
        print(f'{city}: {len(sketch)}')
    start = time.time()
    sample_classification = classifier_helper.classify_samples(test_data_file, output_file, cities_sketches, city_labels, human_sketch, k, seed, M=M, T=threshold)
    print(f'classification: {(time.time()-start)/60} min')


if __name__ == "__main__":
    main("data/training_data.tsv", "data/testing_data.tsv", "run1")
