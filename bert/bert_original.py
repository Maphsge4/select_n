"""
    bert benchmark 叶博代码+profiling
    能跑通，能打印
    TODO: dropout层的参数量为0
"""

# python bert_training.py --training-or-inference=inference

import time
from datetime import timedelta
from typing import List

import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from dummy_dataloader import prepare_dataloader
from torch.optim import SGD, Adam
from lib.transformers import BertConfig,BertForMaskedLM  # lib
from utils import get_parser, seed_all
from validate_old import validate
from train import train


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    seed_all(1)
    device_id = 0
    torch.cuda.set_device(device_id)
    print(f"=> set cuda device = {device_id}")

    config = BertConfig.from_json_file("./bert/bert_large_uncased_config.json")
    model = BertForMaskedLM(config)
    model.cuda()
    print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

    bert_data_loader = prepare_dataloader(2 * args.batch_size, args.batch_size, config.vocab_size)  # 原来是4
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.training_or_inference == "inference":
        validate("bert", model, bert_data_loader, criterion, device_id, args.print_freq)
    else:
        '''
        # resume from checkpoint if one exists;
        state = load_checkpoint(
            args.checkpoint_file, device_id, args.arch, model, optimizer
        )
        '''
        train(bert_data_loader, model, criterion, optimizer, args.epoches, device_id, args.print_freq, args.batch_size)


if __name__ == "__main__":
    time1 = time.time()
    seed_all(1)
    main()
    time2 = time.time()
    time_all = time2 - time1
    print('The total time cost is: {}s'.format(time_all))
