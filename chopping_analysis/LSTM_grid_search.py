import pandas as pd
import json
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Bidirectional
from tensorflow.keras.backend import clear_session
from mfcc_data_prep import prepData




def getData(pd_df, verbose = 1):
    data_dict = prepData(pd_df, 0.005, 7)
    xtrain, ytrain = data_dict['train']
    xtest, ytest = data_dict['test']
    word_code = data_dict['word_code']
    if verbose != 0:
        print(f'x train dim: {xtrain.shape}, y train dim: {ytrain.shape}')
        print(f'x test dim: {xtest.shape}, y test dim: {ytest.shape}')
    return xtrain, ytrain, xtest, ytest



def getConfig():
    nnodes_h1 = [15, 20, 25]
    dropout_h1 = [0.0, 0.2, 0.3]
    nnodes_h2 = [15, 20, 25]
    dropout_h2 = [0.0, 0.2, 0.3]
    merge = ['sum', 'ave']
    nbatch = [32, 48]
    optimiser = ['adam', 'rmsprop']
    nepochs = [20, 30]
    lr = [0.0005, 0.001]
    l2_lambda = [0.0, 0.001]
    configs = []
    for i in nnodes_h1:
        for j in dropout_h1:
            for k in nnodes_h2:
                for l in dropout_h2:
                    for m in merge:
                        for n in nbatch:
                            for o in optimiser:
                                for p in nepochs:
                                    for q in lr:
                                        for r in l2_lambda:
                                            config = (i,j,k,l,m,n,o,p,q,r)
                                            configs.append(config)
    return configs
    


def fitModel(pd_data, cfg):
    xtrain, ytrain, xval, yval = getData(pd_data, verbose = 0)
    nnodes_h1, dropout_h1, nnodes_h2, dropout_h2, merge, nbatch, opt, nepoch, lr, l2_lam = cfg
    nframe = xtrain.shape[1]
    isize = xtrain.shape[2]
    
    model = Sequential()
    model.add(Bidirectional(LSTM(nnodes_h1, return_sequences = True, dropout=dropout_h1,
                            kernel_regularizer=regularizers.L2(l2_lam)),
                            merge_mode=merge, input_shape=(nframe, isize)))
    model.add(Bidirectional(LSTM(nnodes_h2, return_sequences=False, dropout=dropout_h2,
                            kernel_regularizer=regularizers.L2(l2_lam)),
                            merge_mode=merge))
    # model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    # chose optimiser
    if opt == 'adam':
        opt = optimizers.Adam(learning_rate=lr)
    else:
        opt = optimizers.RMSprop(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    history = model.fit(xtrain, ytrain, batch_size=nbatch, validation_data=(xval,yval),
                      epochs=nepoch, verbose=0)
    loss, acc = model.evaluate(xval, yval, verbose=0)
    clear_session()

    return loss, acc, history.history



def gridSearch(data: 'pd DataFrame mfcc data')->'dict averaged model performance for each permutation of hyperparameters':
    configs = getConfig()
    print(len(configs))
    results = {x:{} for x in configs}
    
    for cfg in configs:
        loss = 0
        acc = 0
        loss_epochs = np.zeros(cfg[7])
        acc_epochs = np.zeros(cfg[7])
        loss_val_epochs = np.zeros(cfg[7])
        acc_val_epochs = np.zeros(cfg[7])
        
        for i in range(5):
            current_loss, current_acc, his = fitModel(data, cfg)
            loss+= current_loss
            acc+= current_acc
            loss_epochs+= his['loss']
            acc_epochs+= his['accuracy']
            loss_val_epochs+= his['val_loss']
            acc_val_epochs+= his['val_accuracy']
            
        loss = loss/5
        acc = acc/5
        loss_epochs = loss_epochs/5
        acc_epochs = acc_epochs/5
        loss_val_epochs = loss_val_epochs/5
        acc_val_epochs = acc_val_epochs/5
        
        results[cfg] = {'loss': loss, 'acc': acc, 'loss_epochs': loss_epochs.tolist(),
                        'acc_epochs': acc_epochs.tolist(), 'loss_val_epochs': loss_val_epochs.tolist(),
                        'acc_val_epochs': acc_val_epochs.tolist()}
        print(f'{configs.index(cfg)} - val accuracy: {acc} val loss: {loss}')
    return results

if __name__=='__main__':
    data = pd.read_csv('mfcc_data_centred/S1_v.txt',
                       sep = '\t')
    results = gridSearch(data)
    configs = list(results.keys())
    sterile = {str(x): results[x] for x in configs}
    with open('grid_search_results.json', 'w') as outfile:
        json.dump(sterile, outfile)

