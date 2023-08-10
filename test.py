import json
import os
import pandas as pd
import argparse
import torch
import random
import numpy as np
from config_parser import create_config
from model.models import init_baseline
from tools.train_tool import get_output_folder
from model.CrossCaseCL import CrossCaseCL
from tools.test_tool import test


def init_ours(config='./config/train/train.config', checkpoint=None):
    cross_case_cl = CrossCaseCL(config)

    print('loading from {}'.format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    cross_case_cl.load_state_dict(checkpoint['model'], strict=False)

    return {'encoder_f': cross_case_cl.encoder_f, 'encoder_e': cross_case_cl.encoder_e}


def test_one_checkpoint(config, checkpoint, test_data, device):
    epoch = checkpoint.split('/')[-1].split('.')[0]
    ours = init_ours(config=config, checkpoint=checkpoint)
    outs = test(test_data=test_data, config=config, encoder_f=ours['encoder_f'], encoder_e=ours['encoder_e'],
                device=device, name=epoch)
    return outs


def test_single_file(config, device, test_data, test_data_type, mode='test'):
    # test baselines
    baseline_outputs = dict()
    if config.getboolean('test', 'test_baseline'):
        for i in config.get('test', 'baseline_ids').split(','):
            name = config.get('baseline', 'model{}'.format(i))
            discrete = False
            if i in ['10', '11', '12']:
                discrete = True
                baseline = init_baseline(name)
            else:
                baseline = init_baseline(name, config=config).eval()
            outs = test(test_data=test_data, config=config, encoder_f=baseline,
                        device=device, name=name, is_baseline=True, discrete=discrete)
            baseline_outputs[name] = outs

    # test our checkpoint(s)
    ours_outputs = dict()
    if config.getboolean('test', 'test_ours'):
        test_folder = get_output_folder(config)
    else:
        test_folder = os.path.join(config.get("output", "model_path"), 'baseline', config.get('encoder', 'pooling'))

    if config.getboolean('test', 'test_ours'):
        if config.get('test', 'test_specific') != "None":
            checkpoint = os.path.join(test_folder, config.get('test', 'specific'))
            outs = test_one_checkpoint(config=config, checkpoint=checkpoint, test_data=test_data, device=device)
            epoch = checkpoint.split('/')[-1].split('.')[0]
            ours_outputs['ours-{}'.format(epoch)] = outs
        else:
            for ckpt in os.listdir(test_folder):
                if 'pkl' in ckpt:
                    checkpoint = os.path.join(test_folder, ckpt)
                    epoch = ckpt.split('.')[0]
                    outs = test_one_checkpoint(config=config, checkpoint=checkpoint, test_data=test_data, device=device)
                    ours_outputs['ours-{}'.format(epoch)] = outs

        ours_outputs = dict(sorted(ours_outputs.items(), key=lambda x: x[1][test_level]['avg'], reverse=True))

    # save test results
    pos = config.get('test', 'pos_score')

    excel_out = []
    os.makedirs(os.path.join(test_folder, 'txt_print/'), exist_ok=True)
    with open(os.path.join(test_folder, 'txt_print/test-{}.txt'.format(test_data_type)), 'w', encoding='utf-8') as f:
        f.write('Test folder: {}\n'.format(test_folder))
        f.write('{}-result: \n'.format(test_level))
        f.write('pos_score >= {}\n'.format(pos))
        f.write('model name|' + '|'.join(['{:^8}'.format(m) for m in outs[test_level].keys()]) + '\n')
        f.write('----------' * len(outs[test_level].keys()) + '\n')

        for outputs in [baseline_outputs, ours_outputs]:
            for model_name in outputs:
                print_output = ['{:10}'.format(model_name)]
                excel_out.append({'model': model_name, **outputs[model_name][test_level]})
                for m_name in outputs[model_name][test_level]:
                    score = outputs[model_name][test_level][m_name]
                    print_output.append('{:^8}'.format(str(round(score, 4))))
                f.write('|'.join(print_output) + '\n')

    txt_path = os.path.join(test_folder, 'txt_print/test-{}.txt'.format(test_data_type))
    print('test results saved to {}'.format(txt_path))
    print(open(txt_path).read())

    os.makedirs(os.path.join(test_folder, 'case_study/'), exist_ok=True)
    with open(os.path.join(test_folder, 'case_study/test-{}.json'.format(test_data_type)), 'w') as f:
        json.dump({**baseline_outputs, **ours_outputs}, f, ensure_ascii=False)

    excel_out = pd.DataFrame(excel_out)
    os.makedirs(os.path.join(test_folder, 'excels'), exist_ok=True)
    excel_out.to_excel(os.path.join(test_folder, 'excels/test-{}.xlsx'.format(test_data_type)))
    #
    # if mode == 'dev':
    #     return list(ours_outputs.keys())[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    args = parser.parse_args()

    # set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load test config
    test_config = create_config(args.config)
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_level = 'sent_avg'  # ['sent_avg', 'doc_avg']

    # dev set
    data_type = test_config.get('data', 'valid_data').split('/')[-1].split('.')[0]
    data = list(json.load(open(test_config.get('data', 'valid_data'), encoding='utf-8')).values())
    test_single_file(config=test_config, device=test_device, test_data=data, test_data_type=data_type, mode='dev')

    # test set
    data_type = test_config.get('data', 'test_data').split('/')[-1].split('.')[0]
    data = list(json.load(open(test_config.get('data', 'test_data'), encoding='utf-8')).values())
    test_single_file(config=test_config, device=test_device, test_data=data, test_data_type=data_type)



