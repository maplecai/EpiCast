import selene_sdk
from selene_sdk import samplers, sequences, utils

reference_sequence = sequences.Genome(input_path = '/home/hxcai/ref_genome/hg38.fa', blacklist_regions='hg38')
features = utils.load_features_list('/home/hxcai/cell_type_specific_CRE/data/ATAC_seq/distinct_features.txt')

sampler = samplers.IntervalsSampler(
    reference_sequence = reference_sequence, 
    features = features,
    target_path = '/home/hxcai/cell_type_specific_CRE/data/ATAC_seq/HepG2_ENCFF913MQB.bed',
    intervals_path = '/home/hxcai/cell_type_specific_CRE/data/ATAC_seq/deepsea_TF_intervals.txt',
    sample_negative = True,
    seed = 436,
    validation_holdout = ['chr6', 'chr7'],
    test_holdout = ['chr8', 'chr9'],
    sequence_length = 1000, 
    center_bin_to_predict = 200, 
    feature_thresholds = 0.5, 
    mode = "train", 
    save_datasets = ["test"], 
    output_dir = None)


if __name__ == '__main__':
    x = sampler.sample(batch_size=1)
    print(x[0].shape)

