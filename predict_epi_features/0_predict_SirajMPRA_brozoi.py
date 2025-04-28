import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *

from baskerville.seqnn import SeqNN


class MyBorzoi():
    def __init__(
            self, 
            params_file = "../data/Borzoi/params_pred.json",
            targets_file = "../data/Borzoi/targets_human.txt",
            weights_dir = "../data/Borzoi/weights", 
            reps_num = 1
        ):
        self.reps_num = reps_num
        
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
        for rep_idx in range(reps_num) :
            model_weights = f"{weights_dir}/f{rep_idx}/model0_best.h5"
            self.seqnn_model = SeqNN(model_params)
            self.seqnn_model.restore(model_weights, 0)
            self.seqnn_model.build_slice(target_index)
            # self.seqnn_model.strand_pair.append(slice_pair)
            # self.seqnn_model.build_ensemble(True, [0])
            self.models.append(self.seqnn_model)


    def predict(self, seqs):
        # seqs: (batch_size, 524288, 4)
        pred_list = []

        for rep_idx in range(self.reps_num):
            pred = self.models[rep_idx](seqs)  # (batch_size, out_length, num_targets)

            # 插入rep维度
            pred = pred[:, None, :, :]  # (batch_size, 1, out_length, num_targets)

            # 转成float16
            pred = pred.numpy().astype('float16')  # 注意：这里要从Tensor转成numpy才有astype
            pred_list.append(pred)

        # 把reps拼在一起
        pred_list = np.concatenate(pred_list, axis=1)  # (batch_size, reps_num, out_length, num_targets)
        return pred_list



def get_pred_tf(model, test_data_loader):
    y_pred = []

    for i, batch in enumerate(tqdm(test_data_loader)):
        x = batch.numpy()
        # x = batch['seq'].numpy()  # 注意: 这里把PyTorch tensor转成numpy
        output = model.predict(x)  # 调用 MyBorzoi 的 predict
        y_pred.append(output)

    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred






if __name__ == '__main__':

    data_path = f'data/GosaiMPRA/GosaiMPRA_my_processed_data_len200_norm.csv'
    output_path = f'predict_epi_features/outputs/GosaiMPRA_Borzoi_zero_padding.npy'

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_seed(0)
    

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'cannot find {output_dir}, creating {output_dir}')
    if os.path.exists(output_path):
        print(f'already exists {output_path}, exit')
        exit()
    print(f'predicting {output_path}')


    # dataset = datasets.SeqDataset(
    #     data_path=data_path,
    #     seq_column='seq', 
    #     crop=False,
    #     padding=True,
    #     padding_method='N',
    #     padded_length=196608,
    #     N_fill_value=0)
    #     # padding = False,
    #     # N_fill_value = 0.25,
    
    dataset = datasets.RandomDataset(
        shape=(10, 524288, 4)
    )
    
    model = MyBorzoi()

    test_data_loader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=4)

    pred = get_pred_tf(model, test_data_loader)

    np.save(output_path, pred)



# # MPRA_UPSTREAM  = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
# # MPRA_DOWNSTREAM= 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'
