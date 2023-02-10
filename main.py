import argparse
from config_parser import create_config
import os
from tools.init_tool import init_all
from tools.train_tool import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', default='0', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    args = parser.parse_args()

    gpu_list = []
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        gpu_list = list(range(0, len(args.gpu.split(','))))

    config = create_config(args.config)

    os.system('clear')

    parameters = init_all(config, gpu_list, args.checkpoint, args.seed, mode='train', local_rank=-1)

    train(parameters, config, gpu_list)

