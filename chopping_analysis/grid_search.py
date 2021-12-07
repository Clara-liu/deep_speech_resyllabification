import pandas as pd
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from main_analysis import chop_n_get_acc, prep_data

# results best: config no. 68: [70, 0.05, 60, 0.1, 500, 'ave', 128, 'adam', 80, 0.001]

def getConfig():
    nnodes_h1 = [70, 90]
    dropout_h1 = [0.1, 0.2]
    nnodes_h2 = [40, 60]
    dropout_h2 = [0.1]
    nnodes_dense = [500]
    merge = ['ave']
    nbatch = [64, 128]
    optimiser = ['rmsprop']
    nepochs = [60, 80]
    lr = [0.001, 0.003]
    configs = {}
    config_no = 0
    for i in nnodes_h1:
        for j in dropout_h1:
            for k in nnodes_h2:
                for l in dropout_h2:
                    for m in nnodes_dense:
                        for n in merge:
                            for o in nbatch:
                                for p in optimiser:
                                    for q in nepochs:
                                        for r in lr:
                                                config = (i, j, k, l, m, n, o, p, q, r)
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
        for rep in range(3):
            data = prep_data(df, 0.005, 0.7)
            current_result = chop_n_get_acc(pair_name, config_no, data, config)
            current_result['Rep'] = rep
            if config_no == '0' and rep == 0:
                result = current_result
                result.to_csv(f'grid_search_result_{pair_name}_{data_type}.txt', sep='\t', index=False)
            else:
                saved_result = pd.read_csv(f'grid_search_result_{pair_name}_{data_type}.txt', sep='\t')
                result = pd.concat([saved_result, current_result])
                result.to_csv(f'grid_search_result_{pair_name}_{data_type}.txt', sep='\t', index=False)
            print(f'config no.{config_no}, rep no. {rep} finished')


if __name__ == '__main__':
    grid_search('P1_Onset', 'slow_rate', 'mel_data')