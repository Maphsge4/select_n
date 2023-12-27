from typing import List

import time
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from dummy_dataloader import prepare_dataloader
from lib.my_offload import OffloadModel
from lib.transformers import GPT2Config, GPT2Model
from utils import seed_all, get_parser
from validate import validate
from train import train
from copy import deepcopy

model_name = 'gpt2'
total_iter = 30  # 原来是100

def init_all():
    global args
    device_id = 0
    torch.cuda.set_device(device_id)

    config = GPT2Config.from_json_file("./gpt2/gpt2_large_config.json")
    model = GPT2Model(config=config)
    
    print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

    dataloader = prepare_dataloader(total_iter * args.batch_size, args.batch_size, config.vocab_size)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    return model, config, dataloader, optimizer, criterion


def original(original_model: torch.nn.Module, config, dataloader, optimizer, criterion):
    model = deepcopy(original_model)
    model.cuda()
    model.mode = "original"
    
    return validate(model_name, model, dataloader, criterion, 0, print_freq=1000, preserve_result=True)


def slice(original_model, config, dataloader, optimizer, criterion):
    model = deepcopy(original_model)
    model.mode = "slice"
    mslices = list(model.h)

    model.hh = OffloadModel(
        model=mslices,
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        offload_device=torch.device("cpu"),
        num_slices=10,
        checkpoint_activation=False,
        num_microbatches=1,
    )
    model.to_cuda()
    
    return validate(model_name, model, dataloader, criterion, 0, print_freq=1000, preserve_result=True)


def select(original_model, config, dataloader, optimizer, criterion):
    model = deepcopy(original_model)
    model.mode = "slice"
    mslices = list(model.h)

    model.hh = OffloadModel(
        model=mslices,
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        offload_device=torch.device("cpu"),
        num_slices=10,
        checkpoint_activation=False,
        num_microbatches=1,
        device_list=[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    )
    model.to_cuda()
    
    return validate(model_name, model, dataloader, criterion, 0, print_freq=1000, preserve_result=True)


def compare_result(result_a: List, result_b: List):
    if len(result_a) != len(result_b):
        print("The length of two results are different.")
        return False
    ret = True
    for i in range(len(result_a)):
        if not torch.allclose(result_a[i], result_b[i]):
            print("The {}th result is different.".format(i))
            print(result_a[i], result_b[i])
            ret = False
    return ret


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    original_model, config, dataloader, optimizer, criterion = init_all()

    time1 = time.time()
    seed_all(1)
    slice_result = slice(original_model, config, dataloader, optimizer, criterion)
    time2 = time.time()
    time_all = time2 - time1
    print ('The total time cost is: {}s'.format(time_all))

    time1 = time.time()
    seed_all(1)
    original_result = original(original_model, config, dataloader, optimizer, criterion)
    time2 = time.time()
    time_all = time2 - time1
    print ('The total time cost is: {}s'.format(time_all))

    # time1 = time.time()
    # seed_all(1)
    # select_result = select(original_model, config, dataloader, optimizer, criterion)
    # time2 = time.time()
    # time_all = time2 - time1
    # print ('The total time cost is: {}s'.format(time_all))

    # compare
    print(compare_result(slice_result, original_result))
    # print(compare_result(select_result, original_result))

