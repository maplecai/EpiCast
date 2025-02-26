import sys
sys.path.append("..")
from MPRA_predict.utils import *
from MPRA_predict.datasets import *

from torch.utils.data import DataLoader
# from enformer_pytorch import from_pretrained
from MPRA_predict.models.enformer_pytorch import from_pretrained




def get_pred(model, test_data_loader, device='cuda'):
    model = model.to(device)
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch['seq']
            x = x.to(device)
            output = model(x)
            # if enformer
            if isinstance(output, dict):
                output = output['human']
            y_pred.append(output.detach().cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    torch.cuda.empty_cache()
    return y_pred




def get_pred_total(model, dataset, device, output_path):
    test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    pred = get_pred(model, test_data_loader, device)
    np.save(output_path, pred)
    return



# split to many parts, predict and save, in order to save memory
def get_pred_split(model, dataset, device, output_path, num_splits):

    split_size = len(dataset) // num_splits  # num_splits是你要分割的部分数
    # 分割数据集
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_splits - 1 else len(dataset)

        subset = Subset(dataset, range(start_idx, end_idx))
        subloader = DataLoader(subset, batch_size=4, shuffle=False, num_workers=0)
        y_pred = get_pred(model, subloader, device)
        np.save(f'{output_path}_{i}.npy', np.array(y_pred))
    return




if __name__ == '__main__':

    set_seed(0)
    model_path = f'../pretrained_models/enformer_weights'
    # data_path = f'data/enformer_sequences_test.csv'
    data_path = f'../data/SirajMPRA/SirajMPRA_562654.csv'
    device = f'cuda:0'


    model = from_pretrained(model_path)


    # # no pad, need more modification
    # for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    #     output_path = f'outputs/Enformer_pred_crop_{cropped_length}_pad_196608_no.npy'
    #     if not os.path.exists(output_path):
    #         print(f'predicting {output_path}')
    #         dataset = SeqDataset(
    #             data_path=data_path,
    #             input_column='seq', 
    #             crop=True, ###
    #             crop_method='center',
    #             cropped_length=cropped_length,
    #             padding=False, ###
    #             padding_method='N',
    #             padded_length=196608,
    #             N_fill_value=0.25)
            
    #         test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
    #         pred = get_pred(model, test_data_loader)
    #         pred = pred[:, 447:449]
    #         np.save(output_path, pred)




    # # pad N
    # for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    #     output_path = f'outputs/Enformer_pred_crop_{cropped_length}_pad_196608_N.npy'
    #     if not os.path.exists(output_path):
    #         print(f'predicting {output_path}')
    #         dataset = SeqDataset(
    #             data_path=data_path,
    #             input_column='seq', 
    #             crop=True, ###
    #             crop_method='center',
    #             cropped_length=cropped_length,
    #             padding=True, ###
    #             padding_method='N',
    #             padded_length=196608,
    #             N_fill_value=0.25)
            
    #         test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
    #         pred = get_pred(model, test_data_loader)
    #         pred = pred[:, 447:449]
    #         np.save(output_path, pred)


    # pad zero
    for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
        output_path = f'outputs/SirajMPRA_Enformer_pred_crop_{cropped_length}_pad_196608_N.npy'
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




    # # pad random
    # for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    #     output_path = f'outputs/Enformer_pred_crop_{cropped_length}_pad_196608_random_5_times.npy'
    #     if not os.path.exists(output_path):
    #         print(f'predicting {output_path}')
    #         pred_list = []
    #         for seed in range(5):
    #             set_seed(seed)
    #             dataset = SeqDataset(
    #                 data_path=data_path,
    #                 input_column='seq', 
    #                 crop=True, ###
    #                 crop_method='center',
    #                 cropped_length=cropped_length,
    #                 padding=True, ###
    #                 padding_method='random', ###
    #                 padded_length=196608,
    #                 N_fill_value=0.25)
                
    #             test_data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)
    #             pred = get_pred(model, test_data_loader)
    #             pred = pred[:, 447:449]
    #             pred_list.append(pred)
    #         pred_list = np.stack(pred_list)
    #         np.save(output_path, pred_list)

