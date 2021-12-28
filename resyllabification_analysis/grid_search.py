import pandas as pd
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from main_analysis import get_acc, prep_data


def getConfig():
    nnodes_h1 = [40, 60]
    dropout_h1 = [0.1, 0.2]
    nnodes_h2 = [30, 40]
    dropout_h2 = [0.1, 0.2]
    nnodes_dense = [30, 50]
    merge = ['ave', 'sum']
    nbatch = [16]
    optimiser = ['rmsprop', 'adam']
    nepochs = [70, 85]
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

def grid_search(data_path: 'str path to data file',
                condition: 'str slow or resyllabified',
                rep: 'int how many times to test each config'):
    # read it
    df = pd.read_csv(data_path, sep='\t')
    config_dict = getConfig()
    with open('configs.json', 'w') as file:
        json.dump(config_dict, file)
    for config_no, config in config_dict.items():
        # prep for classifier training
        for i in range(rep):
            data = prep_data(df, 0.005, 0.8)
            _, current_acc = get_acc(config, data)
            current_result = pd.DataFrame([[config_no, i, current_acc]], columns=['Config', 'Rep', 'Accuracy'])
            if config_no == '0' and i == 0:
                current_result.to_csv(f'grid_search_result_{condition}.txt', sep='\t', index=False)
            else:
                saved_result = pd.read_csv(f'grid_search_result_{condition}.txt', sep='\t')
                result = pd.concat([saved_result, current_result])
                result.to_csv(f'grid_search_result_{condition}.txt', sep='\t', index=False)
            print(f'config no.{config_no}, rep no. {i} finished')

if __name__ == '__main__':
    grid_search(f'../pilot_2/mel_data/resyllabified_conosnants/byPair/P0_Onset.txt', 'resyllabified', 5)