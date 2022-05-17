from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from os import listdir
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Masking, Flatten
from tensorflow.keras.backend import clear_session
from tqdm import tqdm

def cat_code(vowel: 'str a or i')-> 'numeric code of category':
    code = 0 if vowel == 'a' else 1
    return code

def prep_data(df: 'pandas mfcc df', interval: 'sampling hop of mfcc df',
              train_ratio, get_diff: 'get velocity' = True) -> 'train/test sets':
    scaler = MinMaxScaler((-10, 20))
    # calculate velocity
    if get_diff:
        diff_df = df.select_dtypes('float').diff(periods=1, axis=0)
        # reset first sample's velocity of each token to 0
        diff_df[df['index'] == 0] = 0
        diff_df = diff_df/interval
        diff_df = scaler.fit_transform(diff_df)
        diff_df = pd.DataFrame(data=diff_df, columns=[str(x) + '_diff' for x in range(diff_df.shape[1])])
        df = pd.concat([df, diff_df], axis=1)
    # group data into unique sequences
    grouped = df.groupby(['Words', 'Rep', 'Speaker'])
    # get longest sequence row number
    longest_len = grouped.agg('count').c0.max()
    # get dimension of feature vectors
    n_feats = df.select_dtypes('float').to_numpy().shape[1]
    # get number of sequences
    n_seqs = len(grouped)
    # initialise X
    X = np.zeros((n_seqs, longest_len, n_feats))
    # initialise Y
    Y = np.zeros((n_seqs, 1))
    # loop through each sequence and pad to max length
    seq_idx = 0
    for name, group in grouped:
        feats = group.select_dtypes('float').to_numpy()
        current_len = feats.shape[0]
        label = cat_code(group['Vowel'].iloc[0])
        X[seq_idx, 0:current_len, :] = feats
        Y[seq_idx,:] = label
        seq_idx += 1
    # randomise features and labels
    p = np.random.permutation(n_seqs)
    X = X[p, :, :]
    Y = Y[p, :]
    # subset training and testing
    ntrain = int(n_seqs*train_ratio)
    x_train = X[:ntrain, :, :]
    y_train = Y[:ntrain, :]
    x_test = X[ntrain:, :, :]
    y_test = Y[ntrain:, :]

    data = {'train': [x_train, y_train], 'test': [x_test, y_test]}
    return data

def get_acc(net_config: 'tuple hyperparameters', data: 'dict train/test sets') -> 'float loss and acc on the test set':
    x_train, y_train = data['train']
    x_test, y_test = data['test']
    nnodes_h1, dropout_h1, nnodes_h2, dropout_h2, nnodes_dense, merge, nbatch, opt, nepoch, lr = net_config
    nframe = x_train.shape[1]
    nfeature = x_train.shape[2]
    net = Sequential()
    net.add(Masking(mask_value=0.0, input_shape= (nframe, nfeature)))
    net.add(Bidirectional(LSTM(nnodes_h1, return_sequences=True, dropout=dropout_h1),
                          merge_mode=merge, input_shape=(nframe, nfeature)))
    net.add(Bidirectional(LSTM(nnodes_h2, return_sequences=False, dropout=dropout_h2),
                          merge_mode=merge))
    #net.add(Flatten())
    net.add(Dense(nnodes_dense, activation='relu'))
    net.add(Dense(15, activation='relu'))
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

def main(data_path: 'str path to subsetted data',
         n_trial: 'int number of times to repeat the model and test',
         config: 'tuple NN configuration'):
    files = listdir(data_path)
    acc_data = []
    for f in files:
        logging.info(f'Processing {f}...')
        # read data
        file_path = f'{data_path}/{f}'
        df = pd.read_csv(file_path, sep = '\t')
        # get pair label
        pair = f.split('_')[0]
        # get condition label: onset or coda
        condition = f.split('_')[1].split('.')[0]
        for i in tqdm(range(n_trial), desc='Repetition trials: '):
            # prep data
            data = prep_data(df, 0.005, 0.8)
            _, acc = get_acc(config, data)
            acc_data_row = [acc, pair, condition, i]
            acc_data.append(acc_data_row)
    result = pd.DataFrame(acc_data, columns=['Accuracy', 'Pair', 'Condition', 'Trial'])
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    result = main('../pilot_2/mel_data/slow_first_cv/byPair', 80, (60, 0.1, 30, 0.2, 50, 'sum', 16, 'adam', 70, 0.001))