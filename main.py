from models import Model
from pytorch_ctc.ctc_ent import ctc_ent_cost
from data_generation import dataProcess, loadData
import torch
import torch.nn.functional as F
import utils
from torch.utils import tensorboard
from torchvision.utils import make_grid

# hyper-parameters and others
params_args = {
    'n_res_cnn': 3,
    'n_rnn': 3,
    'rnn_dim': 512,
    'linear_dim': 300,
    'n_class': 19,
    'n_feats': 128,
    'stride': 1,
    'dropout': 0.1,
    'lr': 0.0001,
    'grad_clip': 400,
    'batch_size': 30,
    'n_epochs': 2,
    'h_rate': 0.1,
    'data_path': 'pilot',
    'use_enctc': False
}
# to monitor training
writer = tensorboard.SummaryWriter('runs/deep_speech_enCTC')

net = Model(params_args['n_res_cnn'], params_args['n_rnn'], params_args['rnn_dim'], params_args['n_class'],
            params_args['n_feats'], params_args['linear_dim'], stride=1, dropout=params_args['dropout'])

train_data, val_data = loadData(params_args['data_path'])


# define optimiser

optimizer = torch.optim.RMSprop(net.parameters(), lr=params_args['lr'])

# train batch function
def train_enctc(train_iter):
    specs, labels, input_lens, label_lens = train_iter.next()
    labels = labels.flatten() # (batch)
    preds = net(specs).transpose(0, 1) # (time, batch, vocab +1)
    H, cost = ctc_ent_cost(preds, labels, input_lens, label_lens)
    cost_total = cost.data.sum()
    inf = float("inf")
    # in case of exploding gradient
    if cost_total == inf or cost_total == -inf or cost_total<=-1e5 or torch.isnan(cost) or torch.isnan(H):
        print("Warning: received an inf loss, setting loss value to 0")
        return torch.zeros(H.size()), torch.zeros(cost.size())
    # set all gradients to zero
    net.zero_grad()
    optimizer.zero_grad()
    (-params_args['h_rate']*H + (1-params_args['h_rate'])*cost).backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), params_args['grad_clip'])
    optimizer.step()

    return H/len(labels), cost/len(labels)

# validation function
def validation_enctc(val_loader):
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    # record average
    val_h_avg = utils.averager()
    val_loss_avg = utils.averager()
    wer_total = 0
    for batch_num, data in enumerate(val_loader):
        specs, labels, input_lens, label_lens = data
        preds = net(specs).transpose(0, 1)
        labels_flat = labels.flatten()
        H, cost = ctc_ent_cost(preds, labels_flat, input_lens, label_lens)
        wer, _ = utils.WER(preds.transpose(0, 1), labels)
        wer_total += wer
        val_h_avg.add(H/len(labels_flat))
        val_loss_avg.add(cost/len(labels_flat))
    val_h = val_h_avg.val()
    val_loss = val_loss_avg.val()
    val_wer = wer_total/len(val_loader)
    val_h_avg.reset()
    val_loss_avg.reset()

    return val_h, val_loss, val_wer


def train(loader, criterion):
    net.train()
    total_loss = 0
    for p in net.parameters():
        p.requires_grad = True
    for batch_idx, data in enumerate(loader):
        specs, labels, input_lens, label_lens = data
        optimizer.zero_grad()
        preds = net(specs)
        preds = F.log_softmax(preds, dim=2).transpose(0, 1)
        loss = criterion(preds, labels, input_lens, label_lens)
        total_loss += loss.item()/len(input_lens)
        loss.backward()
        optimizer.step()
        print(f'batch: {batch_idx} train_loss: {loss.item()/len(input_lens)}')
    return total_loss/len(loader)


def validation(loader, criterion):
    net.eval()
    total_loss = 0
    total_wer = 0
    for p in net.parameters():
        p.requires_grad = False
    for batch_idx, data in enumerate(loader):
        specs, labels, input_lens, label_lens = data
        origin_preds = net(specs)
        preds = F.log_softmax(origin_preds, dim=2).transpose(0, 1)
        loss = criterion(preds, labels, input_lens, label_lens)
        total_loss += loss.item()/len(input_lens)
        wer, _ = utils.WER(origin_preds, labels)
        total_wer += wer
    return total_loss/len(loader), total_wer/len(loader)


def main():
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=params_args['batch_size'],
                                               shuffle=True, collate_fn=lambda x: dataProcess(x))
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=params_args['batch_size'], shuffle=True,
                                         collate_fn=lambda x: dataProcess(x, train=False))
    if params_args['use_enctc']:
        for epoch in range(params_args['n_epochs']):
            # loss averager for each epoch
            h_avg = utils.averager()
            loss_avg = utils.averager()
            train_iter = iter(train_loader)
            for i in range(len(train_loader)):
                for p in net.parameters():
                    p.requires_grad = True
                net.train()
                H, cost = train_enctc(train_iter)
                print(f'epoch {epoch}: batch {i} cost: {cost}')
                # record training loss for each batch
                h_avg.add(H)
                loss_avg.add(cost)
            train_h = h_avg.val()
            train_loss = loss_avg.val()
            h_avg.reset()
            loss_avg.reset()
            val_h, val_loss, val_wer = validation_enctc(val_loader)
            # write to tensorboard
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_h', train_h, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_h', val_h, epoch)
            writer.add_scalar('val_WER', val_wer, epoch)
            print(f'epoch {epoch} wer: {val_wer}')
        writer.close()
    else:
        criterion = torch.nn.CTCLoss(blank=0)
        for epoch in range(params_args['n_epochs']):
            print(f'epoch: {epoch}')
            train_loss = train(train_loader, criterion)
            val_loss, val_wer = validation(val_loader, criterion)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_WER', val_wer, epoch)