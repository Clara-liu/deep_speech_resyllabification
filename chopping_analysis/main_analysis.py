from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from os import listdir
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Flatten
from tensorflow.keras.backend import clear_session
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()



def prep_data(df: 'pandas mfcc df', interval: 'sampling hop of mfcc df',
              train_ratio, data_type='mel_data', get_diff: 'get velocity' = True) -> 'train/test sets':
    # scaler for scaling velocity to same scale as mfcc or mel data
    if data_type == 'mel_data':
        scaler = MinMaxScaler((-5, 20))
    else:
        scaler = MinMaxScaler((-50, 50))
    # calculate velocity
    if get_diff:
        diff_df = df.select_dtypes('float').diff(periods=1, axis=0)
        # reset first sample's velocity of each token to 0
        diff_df[df['index'] == 0] = 0
        diff_df = diff_df/interval
        diff_df = scaler.fit_transform(diff_df)
        diff_df = pd.DataFrame(data=diff_df, columns=[str(x) + '_diff' for x in range(diff_df.shape[1])])
        df = pd.concat([df, diff_df], axis=1)
    # get smallest row count among all tokens
    shortest_len = df.groupby(['Words', 'Rep', 'Speaker']).agg('count').c0.min()
    # trim all tokens to the same length as the shortest one
    df = df[df['index'] < shortest_len]
    # subset train and test set
    # get the features
    X = df.select_dtypes('float').to_numpy()
    # get the number of tokens
    ntokens = int(X.shape[0]/shortest_len)
    # transform dimensions
    X = np.reshape(X, (ntokens, shortest_len, X.shape[1]))
    # convert vowel target column to categorical
    df.Vowel = pd.Categorical(df.Vowel)
    Y = df[df['index'] == 0].Vowel.cat.codes.to_numpy()
    Y = np.reshape(Y, (ntokens, 1))
    # randomise features and labels
    p = np.random.permutation(ntokens)
    X = X[p, :, :]
    Y = Y[p, :]
    # subset training and testing
    ntrain = int(ntokens*train_ratio)
    x_train = X[:ntrain, :, :]
    y_train = Y[:ntrain, :]
    x_test = X[ntrain:, :, :]
    y_test = Y[ntrain:, :]

    data = {'train': [x_train, y_train], 'test': [x_test, y_test]}
    return data



def chop(features: 'np matrix of features (ntoken, n, 30)') ->'features with n-1 less frame (ntoken, n-1, 30)':
    chopped = []
    # loop through all tokens in the feature data
    for token in features:
        chopping_point = token.shape[0] - 1
        token_chopped = token[:chopping_point, :]
        chopped.append(token_chopped)
    chopped = np.array(chopped)
    return chopped


def get_acc(net_config: 'tuple hyperparameters', data: 'dict train/test sets') -> 'float loss and acc on the test set':
    x_train, y_train = data['train']
    x_test, y_test = data['test']
    nnodes_h1, dropout_h1, nnodes_h2, dropout_h2, merge, nbatch, opt, nepoch, lr = net_config
    nframe = x_train.shape[1]
    nfeature = x_train.shape[2]

    net = Sequential()
    net.add(Bidirectional(LSTM(nnodes_h1, return_sequences=True, dropout=dropout_h1),
                            merge_mode=merge, input_shape=(nframe, nfeature)))
    net.add(Bidirectional(LSTM(nnodes_h2, return_sequences=True, dropout=dropout_h2),
                            merge_mode=merge))
    net.add(Flatten())
    net.add(Dense(1, activation='sigmoid'))
    # chose optimiser
    if opt == 'adam':
        opt= optimizers.Adam(learning_rate=lr)
    elif opt == 'sgd':
        opt= optimizers.SGD(learning_rate=lr)
    else:
        opt = optimizers.RMSprop(learning_rate=lr)
    net.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    net.fit(x_train, y_train, batch_size=nbatch, epochs=nepoch, verbose=0)
    loss, acc = net.evaluate(x_test, y_test, verbose=0)
    clear_session()
    return loss, acc


def chop_n_get_acc(pair: 'str pair label', condition: 'str onset or coda',
                   data: 'dict train/test sets', net_config, min_frame: 'int final token length' = 5) -> 'pd dataframe ' \
                    'of test set acc as a function of remaining token length':
    # list for recording rows
    analysis_data = []
    # total number of frames
    total_nframe = data['train'][0].shape[1]
    # how many times to chop
    nchop = total_nframe - min_frame
    # loop through all chops
    for n in range(nchop):
        # calculate remaining duration
        remaining_dur = data['train'][0].shape[1]*0.005
        # get result for full length tokens
        if n == 0:
            loss, acc = get_acc(net_config, data)
        # chop
        else:
            data['train'][0] = chop(data['train'][0])
            data['test'][0] = chop(data['test'][0])
            loss, acc = get_acc(net_config, data)
        # record results
        analysis_data.append([remaining_dur, acc])
    # convert recorded result to pd dataframe
    analysis_data = pd.DataFrame(analysis_data, columns=['Remaining_dur', 'Accuracy'])
    # record various labels
    analysis_data['Pair'] = pair
    analysis_data['Condition'] = condition
    return analysis_data



def analyse(syllabification_condition: 'resyllabified or non_resyllabified or slow',
            ntrial: 'number of times to repeat the chopping analysis',
            config: 'NN hyperparameter config',
            data_type: 'str mfcc_data or mel_data') -> 'pd datafram of chopping analysis':
    # path to folder containiing the minimal pair dfs
    df_folder_path = f'../pilot_2/{data_type}/{syllabification_condition}/byPair'
    # read the file names
    files = listdir(df_folder_path)
    # loop through all the files
    for f in files:
        # det file path
        file_path = f'{df_folder_path}/{f}'
        # read the df
        df = pd.read_csv(file_path, sep = '\t')
        # get pair label
        pair = f.split('_')[0]
        # get condition label: onset or coda
        condition = f.split('_')[1].split('.')[0]
        # loop through all trials
        for i in range(ntrial):
            # get test/train data and randomise
            data = prep_data(df, 0.005, 0.7, data_type=data_type)
            # get result
            current_result = chop_n_get_acc(pair, condition, data, config)
            # record repetition/trial number
            current_result['Rep'] = i
            # concat result dfs
            if i == 0 and files.index(f) == 0:
                result = current_result
            else:
                result = pd.concat([result, current_result])
    result.to_csv(f'chopping_result_{syllabification_condition}_{data_type}.txt', sep='\t', index=False)


if __name__ == '__main__':
    analyse('slow_rate', 10, (45, 0.2, 40, 0.1, 'sum', 64, 'adam', 80, 0.001), 'mel_data')

