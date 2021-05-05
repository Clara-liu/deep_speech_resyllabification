from models import VanillaModel
from data_generation import dataProcess, loadData, LabelConvert
import torch
from torch.utils import tensorboard

# hyper-parameters and others
params_args = {
    'n_res_cnn': 3,
    'n_rnn': 5,
    'rnn_dim': 512,
    'linear_dim': 512,
    'n_class': 16,
    'n_feats': 40,
    'stride': 1,
    'dropout': 0.1,
    'n_convos': 32,
    'lr': 0.0001,
    'batch_size': 32,
    'n_epochs': 85,
    'data_path': 'pilot_1',
}
# to monitor training
writer = tensorboard.SummaryWriter('runs/vanilla_classification_pilot_1_native_gru')
# initiate model
net = VanillaModel(params_args['n_res_cnn'], params_args['n_rnn'], params_args['rnn_dim'], params_args['n_class'],
                   params_args['n_feats'], params_args['linear_dim'], stride=1, dropout=params_args['dropout'],
                   convo_channel=params_args['n_convos'])
# move model to computing unit
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

net.to(device)

# load data
train_data, val_data = loadData(params_args['data_path'], train_tweak_ratio=0.3)

# define optimiser
optimizer = torch.optim.RMSprop(net.parameters(), lr=params_args['lr'])

# initiate label converter
converter = LabelConvert()


def train(loader, error):
    net.train()
    total_loss = 0
    for p in net.parameters():
        p.requires_grad = True
    for batch_idx, data in enumerate(loader):
        specs, labels, _, _ = data
        labels = converter.collapse_seqs(labels)
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = net(specs)
        loss = error(preds, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss/len(loader)


def validation(loader, error):
    net.eval()
    total_loss = 0
    total_accuracy = 0
    for p in net.parameters():
        p.requires_grad = False
    for batch_idx, data in enumerate(loader):
        specs, labels, _, _ = data
        labels = converter.collapse_seqs(labels)
        specs, labels = specs.to(device), labels.to(device)
        preds = net(specs)
        loss = error(preds, labels)
        total_loss += loss.item()
        predictions = torch.argmax(preds, 1)
        ncorrect = (predictions == labels).sum().item()
        accuracy = ncorrect/labels.size()[0]
        total_accuracy += accuracy
    return total_loss/len(loader), total_accuracy/len(loader)



def main():
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=params_args['batch_size'],
                                               shuffle=True, collate_fn=lambda x: dataProcess(x))
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=params_args['batch_size'], shuffle=True,
                                         collate_fn=lambda x: dataProcess(x, train=False))
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(params_args['n_epochs']):
        train_loss = train(train_loader, criterion)
        val_loss, val_accuracy = validation(val_loader, criterion)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_accuracy', val_accuracy, epoch)
        print(f'epoch {epoch}')
    writer.close()


if __name__ == '__main__':
    main()