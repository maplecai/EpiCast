import sys
sys.path.append("..")
from MPRA_predict.utils import *
from MPRA_predict.datasets import *

from torch.utils.data import DataLoader
from MPRA_predict.models.enformer_pytorch import from_pretrained as Enformer_from_pretrained
from MPRA_predict.models.basenji2_pytorch import Basenji2



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
            # if model is enformer
            if isinstance(output, dict):
                output = output['human']
            y_pred.append(output.detach().cpu().numpy()[:, 447:449])
    y_pred = np.concatenate(y_pred, axis=0)
    torch.cuda.empty_cache()
    return y_pred





model_path = '../pretrained_models/basenji2_weights/basenji2.pth'
params_path = '../pretrained_models/basenji2_weights/params_human.json'
with open(params_path) as f:
    model_params = json.load(f)['model']
model = Basenji2(model_params).cuda()
model.load_state_dict(torch.load(model_path), strict=False)


data_path = f'data/enformer_sequences_test.csv'


# # pad zero
# for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
#     output_path = f'outputs/new/Basenji2_pred_crop_{cropped_length}_pad_131072_zero.npy'
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
#             padded_length=131072,
#             N_fill_value=0)
        
#         test_data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
#         pred = get_pred(model, test_data_loader)
#         pred = pred[:, 447:449]
#         np.save(output_path, pred)
#     else:
#         print(f'exist {output_path}, skip')



# # pad N
# for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
#     output_path = f'outputs/new/Basenji2_pred_crop_{cropped_length}_pad_131072_N.npy'
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
#             padded_length=131072,
#             N_fill_value=0.25)
        
#         test_data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
#         pred = get_pred(model, test_data_loader)
#         pred = pred[:, 447:449]
#         np.save(output_path, pred)
#     else:
#         print(f'exist {output_path}, skip')



# pad random
for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    output_path = f'outputs/Basenji2_pred_crop_{cropped_length}_pad_131072_random_5_times.npy'
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
                padded_length=131072,
                N_fill_value=0.25)
            
            test_data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            pred = get_pred(model, test_data_loader)
            # pred = pred[:, 447:449]
            pred_list.append(pred)
        pred_list = np.stack(pred_list)
        np.save(output_path, pred_list)
    else:
        print(f'exist {output_path}, skip')

