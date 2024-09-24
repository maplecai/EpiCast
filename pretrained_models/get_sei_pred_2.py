import sys
sys.path.append('..')
from MPRA_predict.utils import *
from MPRA_predict.datasets import SeqLabelDataset
from MPRA_predict.models import Beluga, Sei

def get_pred(model, test_data_loader, output_file, device='cuda'):
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for (x, _) in tqdm(test_data_loader):
            x = x.to(device)
            x_rc = onehots_reverse_complement(x).to(device)
            pred_1 = model(x)
            pred_2 = model(x_rc)
            pred = (pred_1 + pred_2) / 2
            pred = pred.cpu().detach().numpy()
            with open(output_file, 'ab') as f:
                np.save(f, pred)

trained_model_path = '../pretrained_models/Sei/sei.pth'
model = Sei()
state_dict = torch.load(trained_model_path)
new_state_dict = {k.replace('module.model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

dataset = SeqLabelDataset(seq_exp_path='/home/hxcai/cell_type_specific_CRE/data/AgarwalMPRA/SirajMPRA_total.csv',
                          input_column='seq', padded_len=4096)
test_data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

output_file = '../pretrained_models/Sei/Sei_Siraj_pred_2.npy'
get_pred(model, test_data_loader, output_file)
