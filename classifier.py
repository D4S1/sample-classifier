import classifier_helper

def main():
    ci = 4
    T = 0.5
    M = 8
    k = 24
    s = 1000
    seed = 7921

    hg38_path = ''
    train_path = ''
    hg_38_sequence = classifier_helper.preprocess_human_genome(hg38_path)
    human_kmers = classifier_helper.kmer_set(hg_38_sequence, k, seed, ci)

    cities_sketches = classifier_helper.preprocess_reference(train_path, k, s, human_kmers)