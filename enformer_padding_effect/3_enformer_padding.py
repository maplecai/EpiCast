import sys
sys.path.append("..")
from MPRA_predict.utils import *
from MPRA_predict.datasets import *

from torch.utils.data import DataLoader
from enformer_pytorch import from_pretrained


model_path = f'../pretrained_models/enformer_weights'
data_path = f'data/enformer_sequences_test.csv'
model = from_pretrained(model_path)


# pad N
for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    output_path = f'outputs/Enformer_pred_crop_{cropped_length}_pad_196608_N.npy'
    if not os.path.exists(output_path):
        print(f'predicting {output_path}')
        dataset = SeqDataset(
            data_path=data_path,
            input_column='seq', 
            crop=True, ###
            crop_method='center',
            cropped_length=cropped_length,
            padding=True, ###
            padding_method='N',
            padded_length=196608,
            N_fill_value=0.25)
        
        test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
        pred = get_pred(model, test_data_loader)
        pred = pred[:, 447:449]
        np.save(output_path, pred)


# pad zero
for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    output_path = f'outputs/Enformer_pred_crop_{cropped_length}_pad_196608_N.npy'
    if not os.path.exists(output_path):
        print(f'predicting {output_path}')
        dataset = SeqDataset(
            data_path=data_path,
            input_column='seq', 
            crop=True, ###
            crop_method='center',
            cropped_length=cropped_length,
            padding=True, ###
            padding_method='N',
            padded_length=196608,
            N_fill_value=0)
        
        test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
        pred = get_pred(model, test_data_loader)
        pred = pred[:, 447:449]
        np.save(output_path, pred)


# pad no
for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    output_path = f'outputs/Enformer_pred_crop_{cropped_length}_pad_196608_no.npy'
    if not os.path.exists(output_path):
        print(f'predicting {output_path}')
        dataset = SeqDataset(
            data_path=data_path,
            input_column='seq', 
            crop=True, ###
            crop_method='center',
            cropped_length=cropped_length,
            padding=False, ###
            padding_method='N',
            padded_length=196608,
            N_fill_value=0.25)
        
        test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
        pred = get_pred(model, test_data_loader)
        pred = pred[:, 447:449]
        np.save(output_path, pred)


# pad random
for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    output_path = f'outputs/Enformer_pred_crop_{cropped_length}_pad_196608_random_5_times.npy'
    if not os.path.exists(output_path):
        print(f'predicting {output_path}')
        pred_list = []
        for seed in range(5):
            set_seed(seed)
            dataset = SeqDataset(
                data_path=data_path,
                input_column='seq', 
                crop=True, ###
                crop_method='center',
                cropped_length=cropped_length,
                padding=True, ###
                padding_method='random', ###
                padded_length=196608,
                N_fill_value=0.25)
            
            test_data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)
            pred = get_pred(model, test_data_loader)
            pred = pred[:, 447:449]
            pred_list.append(pred)
        pred_list = np.stack(pred_list)
        np.save(output_path, pred_list)

