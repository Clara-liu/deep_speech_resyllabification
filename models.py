'''
models/modules designed inspired by Deep Speech 2: https://arxiv.org/pdf/1512.02595.pdf,
ResNet: https://arxiv.org/pdf/1603.05027.pdf,
as well as assembly AI: https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO
'''

from torch import nn
from data_generation import loadData, dataProcess

class Norm(nn.Module):
    def __init__(self, n_mels):
        super(Norm, self).__init__()
        self.layer = nn.LayerNorm(n_mels)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer(x)
        return x.transpose(2, 3).contiguous()


class ResNet(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, stride, drop_out, n_mels):
        super(ResNet, self).__init__()
        self.cnn1 = nn.Conv2d(in_chan, out_chan, kernel, stride, padding=kernel[0]//2)
        self.dropout1 = nn.Dropout(drop_out)
        self.layer_norm1 = Norm(n_mels)
        self.cnn2 = nn.Conv2d(out_chan, out_chan, kernel, stride, padding=kernel[0]//2)
        self.dropout2 = nn.Dropout(drop_out)
        self.layer_norm2 = Norm(n_mels)

    def forward(self, x):
        res = x
        x = self.layer_norm1(x)
        x = nn.functional.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = nn.functional.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += res
        return x

class BiGRU(nn.Module):
    def __init__(self, gru_dim, hidden_size, dropout, batch_first):
        super(BiGRU, self).__init__()
        self.GRU = nn.GRU(input_size=gru_dim, hidden_size = hidden_size, num_layers=1,
                          batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(gru_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = nn.functional.gelu(x)
        x, hidden = self.GRU(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, n_res_layers, n_gru_layers, gru_dim, n_class, n_feats, linear_dim, stride=1,
                 dropout=0.1, test=False):
        super(Model, self).__init__()
        self.testing = test
        # initial cnn for feature extraction
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)
        # residual cnns for low level extraction
        self.res_cnn = nn.Sequential(*[ResNet(32, 32, kernel=(5, 5), stride=1, drop_out=dropout, n_mels = n_feats)
                                       for _ in range(n_res_layers)])
        self.dense = nn.Linear(n_feats*32, gru_dim)
        self.bi_gru = nn.Sequential(*[BiGRU(gru_dim=gru_dim if i ==0 else gru_dim*2, hidden_size=gru_dim,
                                            dropout=dropout, batch_first=i==0) for i in range(n_gru_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(gru_dim*2, linear_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(linear_dim, n_class)
        )

    def forward(self, x):
        if self.testing:
            x = self.cnn(x)
            print(f'after first cnn {x.shape}')
            x = self.res_cnn(x)
            print(f'after res net {x.shape}')
            dim = x.size()
            x = x.view(dim[0], dim[1]*dim[2], dim[3])
            x = x.transpose(1,2)
            x = self.dense(x)
            print(f'after dense {x.shape}')
            x = self.bi_gru(x)
            print(f'after rnn {x.shape}')
            x = self.classifier(x)
            print(f'final {x.shape}')
        else:
            x = self.cnn(x)
            x = self.res_cnn(x)
            dim = x.size()
            x = x.view(dim[0], dim[1]*dim[2], dim[3])
            x = x.transpose(1,2)
            x = self.dense(x)
            x = self.bi_gru(x)
            x = self.classifier(x)
            return x


def testModel():
    train, val = loadData('pilot')
    mel_specs, labels, input_lens, label_lens = dataProcess(train)
    test = mel_specs[0].unsqueeze(1)

    model = Model(2, 2, 512, 10, 128, 500, test=True)
    model(test)

if __name__ == '__main__':
    testModel()