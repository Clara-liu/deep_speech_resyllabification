import pandas as pd
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from main_analysis import chop_n_get_acc, prep_data


def getConfig():
    nnodes_h1 = [40, 60, 80]
    dropout_h1 = [0.1, 0.3]
    nnodes_h2 = [50, 70]
    dropout_h2 = [0.1, 0.3]
    merge = ['sum', 'ave']
    nbatch = [64]
    optimiser = ['adam', 'rmsprop']
    nepochs = [45, 65]
    lr = [0.0005, 0.001]
    configs = {}
    config_no = 0
    for i in nnodes_h1:
        for j in dropout_h1:
            for k in nnodes_h2:
                for l in dropout_h2:
                    for m in merge:
                        for n in nbatch:
                            for o in optimiser:
                                for p in nepochs:
                                    for q in lr:
                                            config = (i, j, k, l, m, n, o, p, q)
                                            configs[f'{config_no}'] = config
                                            config_no += 1
    return configs

def grid_search(pair_name: 'str the pair to use for searching',
                condition: 'str slow_rate resyllabified or non_resyllabified',
                data_type: 'str mfcc_data or mel_data'):
    # path for the dataframe
    pair_path = f'../pilot_2/{data_type}/{condition}/byPair/{pair_name}.txt'
    # read it
    df = pd.read_csv(pair_path, sep='\t')
    config_dict = getConfig()
    with open('configs.json', 'w') as file:
        json.dump(config_dict, file)
    for config_no, config in config_dict.items():
        # prep for classifier training
        data = prep_data(df, 0.005, 0.8)
        current_result = chop_n_get_acc(pair_name, config_no, data, config)
        if config_no == '0':
            result = current_result
        else:
            result = pd.concat([result, current_result])
    result.to_csv(f'grid_search_result_{pair_name}_{data_type}.txt', sep='\t', index=False)


if __name__ == '__main__':
    grid_search('P1_Onset', 'slow_rate', 'mel_data')