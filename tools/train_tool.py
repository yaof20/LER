import logging
import torch
import json
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler
import os

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }
    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def get_output_folder(config):
    # simcse configuration
    simcse = "simcse-{}".format('parallel' if config.getboolean('simcse_loss', 'negatives_parallel') else 'cross')
    # if not config.getboolean('contra_loss', 'positive_query'):
    #     simcse = ''
    if not config.getboolean('simcse_loss', 'use'):
        simcse = ''

    # contra configuration
    rm = '_rm-hard{}{}'.format('-attn' if config.getboolean('contra_loss', 'remove_hard_attention') else "",
                               '-qry' if config.getboolean('contra_loss', 'remove_hard_query') else "")
    contra = "contra-{}-{}{}{}{}".format(config.get('contra_loss', 'query'),
                                         int(config.getboolean('contra_loss', 'negatives_attention')),
                                         int(config.getboolean('contra_loss', 'negatives_value')),
                                         int(config.getboolean('contra_loss', 'negatives_query')),
                                         rm if rm != '_rm-hard' else "")
    if not config.getboolean('contra_loss', 'use'):
        contra = ''

    model_name = "{}{}".format(simcse, '-'+contra if simcse else contra)

    # weighted positive
    use_pos_weight = config.getboolean('positive_weight', 'use')

    weight_source = config.get('positive_weight', 'source')  # dot or cos
    weight_type = config.get('positive_weight', 'type')  # sum or norm
    pos_range = config.get('positive_weight', 'range')  # in-case or in-batch
    normalize = config.get('positive_weight', 'normalize')  # soft or hard
    log_sum = "log" if config.getboolean('positive_weight', 'log_sum') else ""

    './data/train/train_harm_record-wo-test.jsonl'
    train_data_path = config.get('data', 'train_data').replace("_record", "")
    train_type = train_data_path[train_data_path.index('_')+1: train_data_path.index('.jsonl')]

    output_path = os.path.join(config.get("output", "model_path"),
                               train_type,
                               'shared' if config.getboolean("encoder", "shared") else "non_shared",
                               config.get('encoder', 'pooling'),
                               config.get('encoder', 'backbone'),
                               "{}{}".format(model_name, '-weighted' if use_pos_weight else ""),
                               "{}-weight".format(weight_source) if use_pos_weight else "",
                               weight_type if use_pos_weight else "",
                               pos_range if use_pos_weight else "",
                               "{}{}".format(normalize, "-log" if log_sum else "") if use_pos_weight else "")
    os.makedirs(output_path, exist_ok=True)
    return output_path


def train(parameters, config, gpu_list, local_rank=-1):
    epoch = config.getint('train', 'epoch')

    output_path = get_output_folder(config)

    trained_epoch = parameters['trained_epoch'] + 1
    model = parameters['model']

    optimizer = parameters['optimizer']
    dataset = parameters['train_dataset']
    global_step = parameters['global_step']

    step_size = config.getint('train', 'step_size')
    save_step = config.getint('train', 'save_step')
    gamma = config.getfloat('train', 'lr_multiplier')
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    grad_accumulate = config.getint('train', 'grad_accumulate')

    logger.info('Training start ... ')

    with open(output_path + '/logging.txt', 'w') as f:
        info = dict()
        info['training_data'] = config.get('data', 'train_data')

        info['encoder'] = dict(backbone=config.get('encoder', 'backbone'),
                               shared=config.get('encoder', 'shared'),
                               pooling=config.get('encoder', 'pooling'))
        info['attention'] = dict(sim_fct=config.get('attention', 'type'),
                                 scale=config.get('attention', 'scale'),
                                 temperature=config.get('attention', 'temperature'))
        info['simcse'] = dict(use=config.get('simcse_loss', 'use'),
                              parallel=config.get('simcse_loss', 'negatives_parallel'),
                              cross=config.get('simcse_loss', 'negatives_cross'),
                              single=config.get('simcse_loss', 'negatives_parallel_single'),
                              temperature=config.get('simcse_loss', 'temperature'))
        info['contra'] = dict(use=config.get('contra_loss', 'use'),
                              query=config.get('contra_loss', 'query'),
                              neg_attention=config.get('contra_loss', 'negatives_attention'),
                              neg_value=config.get('contra_loss', 'negatives_value'),
                              neg_query=config.get('contra_loss', 'negatives_query'))

        info['weighted'] = dict(use=config.get('positive_weight', 'use'),
                                source=config.get('positive_weight', 'source'),
                                type=config.get('positive_weight', 'type'),
                                range=config.get('positive_weight', 'range'),
                                hard_weighted=config.get('positive_weight', 'normalize'),
                                log=config.get('positive_weight', 'log_sum'))

        info['batch_size'] = config.get('train', 'batch_size')
        info['learning_rate'] = config.get('train', 'learning_rate')
        info['epoch'] = config.get('train', 'epoch')
        info['grad_accumulate'] = config.get('train', 'grad_accumulate')

        f.write(json.dumps(info, indent=4, ensure_ascii=False) + '\n')

    for current_epoch in range(trained_epoch, epoch):

        running_loss = defaultdict(int)
        model.train()

        scaler = GradScaler()
        for step, data in enumerate(dataset):
            for key in data:
                if key == 'offset':
                    continue
                for sub_key in data[key]:
                    if isinstance(data[key][sub_key], torch.Tensor):
                        if len(gpu_list) > 0:
                            data[key][sub_key] = data[key][sub_key].cuda()

            if config.getboolean('train', 'fp16'):
                with torch.cuda.amp.autocast():
                    loss = model(data)
            else:
                loss = model(data)

            for key in loss:
                if loss[key] is not None:
                    running_loss[key] += loss[key].item()

            logging_step = config.getint('train', 'logging_step')
            if step % logging_step == logging_step - 1:
                log_tmp = dict()
                for key in running_loss:
                    log_tmp[key] = '{:5}'.format(round(running_loss[key] / logging_step, 4))

                info = 'epoch: {}, step:{}, loss: {}'.format(current_epoch, step + 1, json.dumps(log_tmp))
                with open(output_path + '/logging.txt', 'a') as f:
                    f.write(info + '\n')
                print(info)
                running_loss = defaultdict(int)
            if config.getboolean('train', 'fp16'):
                scaler.scale(loss['overall']).backward()
            else:
                loss['overall'].backward()

            if (step + 1) % grad_accumulate == 0:
                if config.getboolean('train', 'fp16'):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                exp_lr_scheduler.step()

            if global_step % save_step == save_step - 1:
                checkpoint(
                    os.path.join(output_path, "{}-{}k.pkl".format(current_epoch, (global_step + 1) // save_step)),
                    model,
                    optimizer,
                    current_epoch,
                    config,
                    global_step)

            global_step += 1
