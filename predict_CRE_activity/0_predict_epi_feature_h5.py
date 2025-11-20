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

@torch.no_grad()
def get_pred(model, test_data_loader, device='cuda', writer: HDF5Writer=None, flush_every=10):
    model = model.to(device)
    y_pred = []
    model.eval()
    for i, batch in enumerate(tqdm(test_data_loader)):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch['seq']
        else:
            x = batch
        x = x.to(device)
        output = model(x)

        output = output.detach().cpu().numpy()
        writer.append(output)
        if (i+1) % flush_every == 0:
            writer.flush()
    writer.flush()
    writer.close()



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # args.add_argument('-t', '--task', type=str, default=None)
    args.add_argument('-i', '--input_path', type=str, default=None)
    args.add_argument('-o', '--output_path', type=str, default=None)
    args.add_argument('-m', '--model', type=str, default='Sei')
    args.add_argument('-d', '--device', type=str, default='cuda')
    args = args.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model = args.model
    device = args.device

    data_path = input_path

    if output_path is None:
        filename = os.path.basename(input_path).split('.')[0]
        output_path = f'outputs/{filename}_{model}_pred.h5'

    # if task == 'Gosai_MPRA':
    #     data_path = 'data/Gosai_MPRA/Gosai_MPRA_my_processed_data.csv'
    # elif task == 'Gosai_MPRA_designed':
    #     data_path = 'data/Gosai_MPRA/Gosai_MPRA_designed.csv'
    # elif task == 'Agarwal_MPRA':
    #     data_path = 'data/Agarwal_MPRA/Agarwal_MPRA_joint_56k.csv'
    # elif task == 'CAGI5_MPRA':
    #     data_path = 'data/CAGI5_MPRA/CAGI5_MPRA.csv'
    # elif task == 'Reddy_MPRA':
    #     data_path = 'data/Reddy_MPRA/Reddy_MPRA.csv'
    # elif task == 'Tewhey_MPRA_LCL':
    #     data_path = 'data/Tewhey_MPRA/Tewhey_MPRA.csv'
    # elif task == 'Tewhey_MPRA_Jurkat':
    #     data_path = 'data/Tewhey_Lab_MPRA/Jurkat.csv'
    # elif task == 'eQTL':
    #     data_path = 'data/GTEx_eQTL/Enformer_processed/Whole_Blood_data.csv'
    # elif task == 'cCRE_5_cell_types':
    #     data_path = 'data/cCRE/cCRE_5_cell_types.csv'
    # elif task == 'cCRE_total':
    #     data_path = 'data/cCRE/cCRE_total.csv'
    # elif task == 'random_200bp_seqs':
    #     data_path = 'data/Gosai_MPRA/random_200bp_seqs.csv'
    # elif task == 'random_genome_200bp_seqs':
    #     data_path = 'data/Gosai_MPRA/random_genome_200bp_seqs.csv'
    # else:
    #     raise ValueError(f'task name = {task} not found')

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path):
        print(f'already exists {output_path}, exit')
        exit()
    else:
        print(f'predicting {output_path}')

    set_seed(0)

    if args.model == 'Sei':

        model_path = f'../data/Sei/resources/sei.pth'
        model_state_dict = torch.load(model_path)
        model_state_dict = {k.replace('module.model.', ''): v for k, v in model_state_dict.items()}
        model = models.Sei()
        model.load_state_dict(model_state_dict, strict=False)
        model = model.to(device)

        dataset = datasets.SeqDataset(
            data_path=data_path,
            seq_column='seq', 

            # crop=False,
            crop=True,
            cropped_length=200,

            padding=True,
            padding_method='N',
            padded_length=4096,
            N_fill_value=0.25,
        )

        test_data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
        writer = HDF5Writer(output_path)
        pred = get_pred(model, test_data_loader, device, writer, 512)


    elif args.model == 'Enformer':

        model_path = f'../pretrained_models/enformer_weights'
        model = models.enformer_pytorch.from_pretrained(model_path, target_length=2, use_tf_gamma=False)
        model = model.to(device)

        dataset = datasets.SeqDataset(
            data_path=data_path,
            seq_column='seq', 
            crop=False,
            padding=True,
            padding_method='N',
            padded_length=256,
            # padded_length=196608,
            N_fill_value=0.25,
        )
        
        test_data_loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
        writer = HDF5Writer(output_path)
        pred = get_pred(model, test_data_loader, device, writer, 1024)


    else:
        raise ValueError(f'model name = {args.model} not found')
    # TO DO


    # long long context
    # MPRA_UPSTREAM   = 'CGCGTTGCTGGCGTTTTTCCATAGGCTCCGCCCCCCTGACGAGCATCACAAAAATCGACGCTCAAGTCAGAGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCCTGGAAGCTCCCTCGTGCGCTCTCCTGTTCCGACCCTGCCGCTTACCGGATACCTGTCCGCCTTTCTCCCTTCGGGAAGCGTGGCGCTTTCTCATAGCTCACGCTGTAGGTATCTCAGTTCGGTGTAGGTCGTTCGCTCCAAGCTGGGCTGTGTGCACGAACCCCCCGTTCAGCCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGTAAGACACGACTTATCGCCACTGGCAGCAGCCACTGGTAACAGGATTAGCAGAGCGAGGTATGTAGGCGGTGCTACAGAGTTCTTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGAACAGTATTTGGTATCTGCGCTCTGCTGAAGCCAGTTACCTTCGGAAAAAGAGTTGGTAGCTCTTGATCCGGCAAACAAACCACCGCTGGTAGCGGTGGTTTTTTTGTTTGCAAGCAGCAGATTACGCGCAGAAAAAAAGGATCTCAAGAAGATCCTTTGATCTTTTCTACGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGCGGCCGCAAATGCTAAACCACTGCAGTGGTTACCAGTGCTTGATCAGTGAGGCACCGATCTCAGCGATCTGCCTATTTCGTTCGTCCATAGTGGCCTGACTCCCCGTCGTGTAGATCACTACGATTCGTGAGGGCTTACCATCAGGCCCCAGCGCAGCAATGATGCCGCGAGAGCCGCGTTCACCGGCCCCCGATTTGTCAGCAATGAACCAGCCAGCAGGGAGGGCCGAGCGAAGAAGTGGTCCTGCTACTTTGTCCGCCTCCATCCAGTCTATGAGCTGCTGTCGTGATGCTAGAGTAAGAAGTTCGCCAGTGAGTAGTTTCCGAAGAGTTGTGGCCATTGCTACTGGCATCGTGGTATCACGCTCGTCGTTCGGTATGGCTTCGTTCAACTCTGGTTCCCAGCGGTCAAGCCGGGTCACATGATCACCCATATTATGAAGAAATGCAGTCAGCTCCTTAGGGCCTCCGATCGTTGTCAGAAGTAAGTTGGCCGCGGTGTTGTCGCTCATGGTAATGGCAGCACTACACAATTCTCTTACCGTCATGCCATCCGTAAGATGCTTTTCCGTGACCGGCGAGTACTCAACCAAGTCGTTTTGTGAGTAGTGTATACGGCGACCAAGCTGCTCTTGCCCGGCGTCTATACGGGACAACACCGCGCCACATAGCAGTACTTTGAAAGTGCTCATCATCGGGAATCGTTCTTCGGGGCGGAAAGACTCAAGGATCTTGCCGCTATTGAGATCCAGTTCGATATAGCCCACTCTTGCACCCAGTTGATCTTCAGCATCTTTTACTTTCACCAGCGTTTCGGGGTGTGCAAAAACAGGCAAGCAAAATGCCGCAAAGAAGGGAATGAGTGCGACACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
    # MPRA_DOWNSTREAM = 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTGACCTACGGCGTGCAGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAAGGTCTATATCACCGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGACCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAATGATAATAATTCTAGAGTCGGGGCGGCCGGCCGCTTCGAGCAGACATGATAAGATACATTGATGAGTTTGGACAAACCACAACTAGAATGCAGTGAAAAAAATGCTTTATTTGTGAAATTTGTGATGCTATTGCTTTATTTGTAACCATTATAAGCTGCAATAAACAAGTTAACAACAACAATTGCATTCATTTTATGTTTCAGGTTCAGGGGGAGGTGTGGGAGGTTTTTTAAAGCAAGTAAAACCTCTACAAATGTGGTAAAATCGATAAGGATCCGTCGACCGATGCCCTTGAGAGCCTTCAACCCAGTCAGCTCCTTCCGGTGGGCGCGGGGCATGACTATCGTCGCCGCACTTATGACTGTCTTCTTTATCATGCAACTCGTAGGACAGGTGCCGGCAGCGCTCTTCCGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGTTCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATACGGTTATCCACAGAATCAGGGGATAACGCAGGAAAGAACATGTGAGCAAAAGGCCAGCAAAAGGCCAGGAACCGTAAAAAGGC'

    # 300bp context
    # MPRA_UPSTREAM   = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
    # MPRA_DOWNSTREAM = 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'
