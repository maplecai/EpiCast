import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import h5py

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from baskerville.seqnn import SeqNN





class MyBorzoi():
    def __init__(
            self, 
            params_file = "./data/Borzoi/my_params_crop_256.json",
            targets_file = "./data/Borzoi/targets_human.txt",
            weights_dir = "./data/Borzoi/weights", 
            num_reps = 1
        ):
        self.num_reps = num_reps
        
        with open(params_file) as f:
            params = json.load(f)
        model_params = params['model']

        self.targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
        target_index = self.targets_df.index # output channels
        # strand_pair = self.targets_df.strand_pair
        # target_slice_dict = {ix : i for i, ix in enumerate(target_index.values.tolist())}
        # slice_pair = np.array([
        #     target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()
        #     ], dtype='int32') # output strand index
        
        self.models = []
        for rep_idx in range(num_reps) :
            model_weights = f"{weights_dir}/f{rep_idx}/model0_best.h5"
            self.model = SeqNN(model_params)
            self.model.restore(model_weights, 0)
            # self.seqnn_model.build_slice(target_index)
            # self.seqnn_model.strand_pair.append(slice_pair)
            # self.seqnn_model.build_ensemble(True, [0])
            self.models.append(self.model)


    def predict(self, seqs):
        # seqs: (batch_size, 524288, 4)
        pred_list = []

        for rep_idx in range(self.num_reps):
            pred = self.models[rep_idx](seqs)  # (batch_size, out_length, num_targets)
            pred = pred[:, None, :, :]  # (batch_size, num_reps, out_length, num_targets)
            pred_list.append(pred)

        # 把reps拼在一起
        pred_list = np.concatenate(pred_list, axis=1)  # (batch_size, num_reps, out_length, num_targets)
        return pred_list



def get_pred_tf(model, test_data_loader, writer: H5BatchWriter, flush_every=1):
    y_pred = []
    for i, batch in enumerate(tqdm(test_data_loader)):
        if isinstance(batch, tuple):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch['seq']
        elif isinstance(batch, torch.Tensor):
            x = batch
        x = x.numpy()  # 把PyTorch tensor转成numpy
        output = model.predict(x)
        y_pred.append(output)

        writer.save(output)
        if (i+1) % flush_every == 0:
            writer.flush()
    writer.flush()
    writer.close()
    return






if __name__ == '__main__':

    data_path = f'data/Gosai_MPRA/Gosai_MPRA_my_processed_data_len200_norm.csv'
    output_path = f'predict_epi_features/outputs/Gosai_MPRA_Borzoi_pred.h5'

    set_seed(0)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'cannot find {output_dir}, creating {output_dir}')
    if os.path.exists(output_path):
        print(f'already exists {output_path}, exit')
        exit()
    print(f'predicting {output_path}')

    model = MyBorzoi()

    dataset = datasets.SeqDataset(
        data_path=data_path,
        seq_column='seq', 
        crop=False,
        padding=True,
        padding_method='N',
        padded_length=256,
        N_fill_value=0,
    )

    test_data_loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=8)

    writer = H5BatchWriter(output_path, "pred", dtype=np.float16)

    pred = get_pred_tf(model, test_data_loader, writer)

    # np.save(output_path, pred)



# # MPRA_UPSTREAM  = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
# # MPRA_DOWNSTREAM= 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'
