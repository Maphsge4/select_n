"""
    gpt2 benchmark 叶博代码+profiling
    能跑通，能打印
    TODO: dropout层的参数量为0
"""
from typing import List

import time
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from dummy_dataloader import prepare_dataloader
from lib.my_offload import OffloadModel
from lib.transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
# from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
from utils import seed_all, get_parser
from gene import validate
from train import train

model_name = 'gpt2'


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    seed_all(1)
    device_id = 0
    torch.cuda.set_device(device_id)
    config = GPT2Config.from_json_file("./gpt/gpt2_large_config.json")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    # model = GPT2LMHeadModel(config)
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if args.mode == "original":
        model.transformer.mode = "original"

        model.cuda(device_id)
        print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

    elif args.mode == "slice":
        model.transformer.mode = "slice"

        mslices : List[nn.Module] = []
        for i, layer_module in enumerate(model.transformer.h):  # 12层
            mslices.append(layer_module)

        model.transformer.hh = OffloadModel( # 使用 OffloadModel 来包装模型
            name="gpt2", # 模型名称
            model=mslices, # 原生模型
            device=torch.device("cuda"), # 用于计算向前和向后传播的设备
            offload_device=torch.device("cpu"), # 模型将存储在其上的offload 设备
            num_slices=10, # 模型应分片的片数
            checkpoint_activation=False,
            num_microbatches=1,
        )
        model.transformer.to_cuda()   # only load embeddings to cuda

    # inputs = tokenizer(["An increasing sequence: one,"], return_tensors="pt")
    print("max:", torch.cuda.max_memory_allocated(device=torch.device("cuda")))  # 显存量
    print("now", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量

    dataloader = prepare_dataloader(args.batch_size, args.batch_size, 50257)  # 原来是4
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    if args.training_or_inference == "inference":
        validate(model_name, model, dataloader, criterion, device_id, print_freq=1000)
    else:
        '''
        # resume from checkpoint if one exists;
        state = load_checkpoint(
            args.checkpoint_file, device_id, args.arch, model, optimizer
        )
        '''
        train(dataloader, model, criterion, optimizer, args.epoches, device_id, args.print_freq, args.batch_size)  


if __name__ == "__main__":
    time1 = time.time()
    seed_all(1)
    main()
    time2 = time.time()
    time_all = time2 - time1
    print ('The total time cost is: {}s'.format(time_all))