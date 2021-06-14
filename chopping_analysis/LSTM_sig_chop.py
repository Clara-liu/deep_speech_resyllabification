import pandas as pd
import numpy as np

from os import listdir
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from LSTM_grid_search import getData
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Bidirectional
from tensorflow.keras.backend import clear_session
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()



def getFileNames(path:'path to mfcc data files')->'list data paths':
    paths = listdir(path)
    return paths

def chopData(np_features: 'np dataframe mfcc features')->'np features with n-1 frames':
    chopped = []
    for seq in np_features:
        nframe = seq.shape[0]
        seq_chopped = seq[:(nframe-1), :]
        chopped.append(seq_chopped)
    chopped = np.array(chopped)
    return chopped


def getAcc(model_config: 'tuple model hyperparameters', 
           data: 'tuple train and test data')->'float loss and acc on test data':
    xtrain, ytrain, xval, yval = data

    nnodes_h1, dropout_h1, nnodes_h2, dropout_h2, merge, nbatch, opt, nepoch, lr = model_config
    nframe = xtrain.shape[1]
    isize = xtrain.shape[2]
    
    model = Sequential()
    model.add(Bidirectional(LSTM(nnodes_h1, return_sequences = True, dropout= dropout_h1),
                                merge_mode = merge, input_shape = (nframe, isize)))
    model.add(Bidirectional(LSTM(nnodes_h2, return_sequences = False, dropout = dropout_h2),
                            merge_mode = merge))
    # model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    # chose optimiser
    if opt == 'adam':
        opt= optimizers.Adam(learning_rate = lr)
    elif opt == 'sgd':
        opt= optimizers.SGD(learning_rate = lr)
    else:
        opt = optimizers.RMSprop(learning_rate = lr)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    
    model.fit(xtrain, ytrain, batch_size=nbatch, epochs= nepoch, verbose = 0)
    loss, acc = model.evaluate(xval, yval, verbose = 0)
    clear_session()
    return loss, acc



def chopAnalyse(triplet: 'str triplet', contrast: 'str v or c pair',
             pd_data: 'pd dataframe mfcc dataframe', config: 'tuple hyperparameters',
             min_frame: 'int final chopped seq length' = 5,
             nrep: 'int number of random subset and test'= 10)->'pd dataframe test acc with running chop':
    data = []
    count = 0
    for i in range(nrep):
        xtrain, ytrain, xtest, ytest = getData(pd_data)
        total_frame = xtrain.shape[1]
        nchop = total_frame-min_frame
        if i == 0:
            print(f'total networks: {nrep*nchop}')
        for n in range(nchop):
            seq_dur = (total_frame-1-n)*0.005
            if n == 0:
                _, acc = getAcc(config, (xtrain, ytrain, xtest, ytest))
            else:
                xtrain = chopData(xtrain)
                xtest = chopData(xtest)
                _, acc = getAcc(config, (xtrain, ytrain, xtest, ytest))
            data.append([i, seq_dur, acc])
            print(f'current network number: {count+1} accuracy {acc}')
            count+= 1
    data = pd.DataFrame(data, columns = ['Rep', 'Seq_dur', 'Accuracy'])
    data['Triplet'] = triplet
    data['Contrast'] = contrast
    return data

def analyse(files: 'list of str file names', ntriplet: 'int number of triplets to analyse', config,
            dir_path: 'str directory to files')-> None:
    files = [x for x in files if (('DS' not in x) and (int(re.sub("[^0-9]", "", x)) <= ntriplet))]
    for f in files:
        print(f'current file: {f}')
        triplet, contrast = f.strip('.txt').split('_')
        file = pd.read_csv(f'{dir_path}/{f}', sep = '\t')
        current = chopAnalyse(triplet, contrast, file, config, min_frame = 5, nrep = 10)
        if files.index(f) == 0:
            result = current
        else:
            result = pd.concat([result, current])
    result.to_csv(f'results_{ntriplet}.txt', sep = '\t', index = False)

if __name__=='__main__':
    config = (15, 0.2, 20, 0.3, 'sum', 32, 'adam', 30, 0.001)
    file_dir = 'C:\\Users\zirui.liu\Documents\chop_analysis_LSTM\mfcc_data\winlen_25\mfcc_data_centred'
    file_paths = getFileNames(file_dir)
    analyse(file_paths, 9, config, file_dir)
