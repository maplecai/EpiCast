import sys
sys.path.append('..')
from MPRA_predict.utils import *
from MPRA_predict.datasets import SeqLabelDataset
from MPRA_predict.models import Beluga

def get_pred(model, test_data_loader, device='cuda'):
    model = model.to(device)
    y_pred = []
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(test_data_loader):
            x = x.to(device)
            x_rc = onehots_reverse_complement(x).to(device)
            pred_1 = model(x)
            pred_2 = model(x_rc)
            pred = (pred_1 + pred_2) / 2
            y_pred.append(pred.cpu().detach().numpy())
            break
    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred


model = Beluga()
trained_model_path = '../pretrained_models/Beluga/deepsea.beluga.pth'
state_dict = torch.load(trained_model_path)
new_state_dict = {k.replace('module.model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

dataset = SeqLabelDataset(seq_exp_path='/home/hxcai/cell_type_specific_CRE/data/SirajMPRA/SirajMPRA_total.csv',
                          input_column='seq', padded_len=2000)
# test_data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
# y_pred = get_pred(model, test_data_loader)
# np.save(f'../pretrained_models/Beluga/Beluga_Siraj_pred_b64.npy', y_pred)



# 使用不同的 batch_size 进行推理,发现结果不同
batch_sizes = [1, 32, 64, 128]
for batch_size in batch_sizes:
    test_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    y_pred = get_pred(model, test_data_loader)
    np.save(f'../pretrained_models/Beluga/Beluga_Siraj_pred_b{batch_size}.npy', y_pred)