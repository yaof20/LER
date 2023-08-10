import logging
from dataset.data_processor import CLProcessor
from transformers import BertTokenizerFast, AutoTokenizer
from torch.utils.data import DataLoader
from model.CrossCaseCL import CrossCaseCL
from model.optimizer import init_optimizer
from model.utils import init_tokenizer
import random
import torch
import numpy as np

logger = logging.getLogger(__name__)


def block_seed(seed, gpu_list):
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if len(gpu_list) > 0:
        torch.cuda.manual_seed_all(seed)
    

def init_all(config, gpu_list, checkpoint, seed, mode, *args, **params):
    # prepare parameters
    parameters = {}
    
    # set random seed to a fixed one
    block_seed(seed, gpu_list)
    
    # load dataset
    tokenizer = init_tokenizer(config.get('encoder', 'backbone'))
    dataset = CLProcessor(config, tokenizer, input_file=config.get('data', 'train_data'))
    train_dataset = DataLoader(dataset,
                               batch_size=config.getint('train', 'batch_size'),
                               collate_fn=dataset.collate_fn,
                               shuffle=True)
    # load model
    model = CrossCaseCL(config)
    if gpu_list:
        model = model.cuda()
    
    # training parameters from scratch
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = -1
    global_step = 0
    
    # load checkpoint if applicable
    try:
        existing = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(existing['model'])
        
        if mode == 'train':
            trained_epoch = existing['trained_epoch']
            if config.get('train', 'optimizer') == existing['optimizer_name']:
                optimizer.load_state_dict(existing['optimizer'])
            else:
                logger.warning('optimizer changed, do not load parameters of existing optimizer checkpoint')
            if 'global_step' in existing:
                global_step = existing['global_step']
                
    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            logger.error(information)
            raise e
        else:
            logger.warning(information)

    parameters['model'] = model
    
    if mode == 'train':
        parameters['train_dataset'] = train_dataset
        parameters['optimizer'] = optimizer
        parameters['trained_epoch'] = trained_epoch
        parameters['global_step'] = global_step
    
    logger.info('Initialize done.')

    return parameters
