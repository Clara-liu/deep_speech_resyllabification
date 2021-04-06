from models import Model
from pytorch_ctc.ctc_ent import ctc_ent_cost
from data_generation import dataProcess, loadData
import torch
import utils
from torch.utils import tensorboard
from torchvision.utils import make_grid

# hyper-parameters and others
params_args = {
    'n_res_cnn': 3,
    'n_rnn': 5,
    'rnn_dim': 512,
    'linear_dim': 512,
    'n_class': 19,
    'n_feats': 128,
    'stride': 1,
    'dropout': 0.1,
    'lr': 0.00005,
    'grad_clip':400,
    'batch_size': 50,
    'n_epochs': 20,
    'h_rate': 0.1,
    'data_path': 'pilot'
}
# to monitor training
writer = tensorboard.SummaryWriter('runs/deep_speech_enCTC')

net = Model(params_args['n_res_cnn'], params_args['n_rnn'], params_args['rnn_dim'], params_args['n_class'],
            params_args['n_feats'], params_args['linear_dim'], stride=1, dropout=params_args['dropout'])

train_data, val_data = loadData(params_args['data_path'])

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=params_args['batch_size'], shuffle=True,
                                           collate_fn=lambda x: dataProcess(x))
val_loader = torch.utils.data.DataLoader(dataset=val_data, shuffle=True,
                                         collate_fn=lambda x: dataProcess(x, train=False))
# loss averager within batches
h_avg = utils.averager()
loss_avg = utils.averager()

# define optimiser

optimizer = torch.optim.RMSprop(net.parameters(), lr=params_args['lr'])

# train batch function
def train():
    specs, labels, input_lens, label_lens = train_iter.next()
    labels = labels.flatten()
    preds = net(specs).transpose(0, 1)
    H, cost = ctc_ent_cost(preds, labels, input_lens, label_lens)
    cost_total = cost.data.sum()
    inf = float("inf")
    # in case of exploding gradient
    if cost_total == inf or cost_total == -inf or cost_total<=-1e5 or torch.isnan(cost) or torch.isnan(H):
        print("Warning: received an inf loss, setting loss value to 0")
        return torch.zeros(H.size()), torch.zeros(cost.size())
    # set all gradients to zero
    net.zero_grad()
    (-params_args['h_rate']*H + (1-params_args['h_rate'])*cost).backward()
    torch.nn.utils.clip_grad_norm(net.parameters(), params_args['grad_clip'])
    optimizer.step()
    return H/params_args['batch_size'], cost/params_args['batch_size']