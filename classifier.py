import classifier_helper
import utils
import time

def main(train_data_file: str, test_data_file: str, output_file: str, k: int=24, s: int=10000, ci: int = 4, seed=12345, M: int = 3, T: float = 0.5):

    human_sketch = utils.load_pickle('human_sketch.pkl')
    
    
    cities_sketches = classifier_helper.preprocess_reference(train_data_file, k=k, s=s, human_set = human_sketch, seed=seed, ci=ci)
    start = time.time()
    sample_classification = classifier_helper.classify_samples(test_data_file, output_file, cities_sketches, human_sketch, k, s, seed, ci)
    print(f'reference prep: {(time.time()-start)/60} min')


if __name__ == "__main__":
    main("data/training_data.tsv", "data/testing_data.tsv", "run1")
