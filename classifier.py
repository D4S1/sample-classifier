import classifier_helper

def main():
    hg38_path = ''
    hg_38_sequence = classifier_helper.preprocess_human_genome(hg38_path)
    human_kmers = classifier_helper.kmer_set(hg_38_sequence)

    cities_sketches = classifier_helper.preprocess_reference()