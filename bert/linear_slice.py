"""
    bert + offload_model 叶博代码+profiling
    能跑通，能打印
    TODO: dropout层的参数量为0
"""

# python bert_training.py --training-or-inference=inference

import time
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
from lib.my_offload import OffloadModel
from lib.transformers import BertConfig, BertForMaskedLM
from torch.optim import SGD, Adam
from utils import seed_all, get_parser
from validate_old import validate
from train import train


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    seed_all(1)
    device_id = 0
    torch.cuda.set_device(device_id)
    print(f"=> set cuda device = {device_id}")

    # 定义模型
    l_model = nn.Sequential(
        nn.Identity(),  # 输入层
        nn.Linear(512, 2048),  # 输入层到隐藏层1
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 2048),  # 隐藏层1到隐藏层2
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 2048),  # 隐藏层2到隐藏层3
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 2048),  # 隐藏层3到隐藏层4
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 2048),  # 隐藏层4到隐藏层5
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 2048),  # 隐藏层5到隐藏层6
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 2048),  # 隐藏层6到隐藏层7
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 2048),  # 隐藏层7到隐藏层8
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 2048),  # 隐藏层8到隐藏层9
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 2048),  # 隐藏层9到隐藏层10
        nn.ReLU(),  # 激活函数
        nn.Linear(2048, 10),  # 隐藏层10到输出层
        nn.Softmax(dim=1)  # 输出层的softmax激活函数
    )
    
    model= OffloadModel(
        model=l_model, # 原生模型
        device=torch.device("cuda"), # 用于计算向前和向后传播的设备
        offload_device=torch.device("cpu"), # 模型将存储在其上的offload 设备
        num_slices=23, # 模型应分片的片数
        checkpoint_activation=False,
        num_microbatches=1,
    )
    print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

    bert_data_loader = prepare_dataloader(2 * args.batch_size, args.batch_size, 30522)  # 原来是4
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(l_model.parameters(), lr=0.01)

    validate("linear", model, bert_data_loader, criterion, device_id, args.print_freq)


if __name__ == "__main__":
    time1 = time.time()
    seed_all(1)
    main()
    time2 = time.time()
    time_all = time2 - time1
    print ('The total time cost is: {}s'.format(time_all))
