from main_ctc import main as search
import json

# search space
setups = {
    'n_res_cnn': [3, 5],
    'n_rnn': [5, 7],
    'rnn_dim': [512, 768],
    'linear_dim': [512, 768],
    'n_class': 6,
    'n_feats': 40,
    'stride': 1,
    'dropout': [0.1, 0.2],
    'n_convos': [16, 32],
    'lr': [0.00005, 0.00001],
    'grad_clip': 400,
    'batch_size': [8, 16],
    'n_epochs': 250,
    'h_rate': [0.0, 0.2],
    'data_path': 'pilot_0',
    'use_enctc': True,
    'blank': None,
    'tweak': [0.3, 0.5]
}

def grid_search(parameters):
    search_count = 0
    search_dict = {}
    for i in parameters['n_res_cnn']:
        for j in parameters['n_rnn']:
            for k in parameters['rnn_dim']:
                for l in parameters['linear_dim']:
                    for m in parameters['dropout']:
                        for n in parameters['n_convos']:
                            for o in parameters['lr']:
                                for p in parameters['batch_size']:
                                    for q in parameters['h_rate']:
                                        for r in parameters['tweak']:
                                            current_setup = {
                                                'n_res_cnn': i,
                                                'n_rnn': j,
                                                'rnn_dim': k,
                                                'linear_dim': l,
                                                'n_class': 6,
                                                'n_feats': 40,
                                                'stride': 1,
                                                'dropout': m,
                                                'n_convos': n,
                                                'lr': o,
                                                'grad_clip': 400,
                                                'batch_size': p,
                                                'n_epochs': 250,
                                                'h_rate': q,
                                                'data_path': 'pilot_0',
                                                'use_enctc': True,
                                                'blank': None,
                                                'tweak': r
                                            }
                                            search_dict[str(search_count)] = current_setup
                                            with open('search_dict.json', 'w') as f:
                                                json.dump(search_dict, f)
                                            search(current_setup, str(search_count))
                                            search_count += 1

if __name__ == '__main__':
    grid_search(setups)