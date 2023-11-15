from typing import List

import time
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from dummy_dataloader import prepare_dataloader
from lib.my_offload import OffloadModel as offload_model
from lib.select_n import OffloadModel as select_model
from lib.transformers import BertConfig, BertForMaskedLM
from utils import seed_all, get_parser
from validate import validate
from train import train
from copy import deepcopy

model_name = 'bert'
total_iter = 3  # 原来是100

def init_all():
    global args
    device_id = 0
    torch.cuda.set_device(device_id)

    config = BertConfig.from_json_file("./bert/bert_large_uncased_config.json")
    model = BertForMaskedLM(config)
    model.bert.mode = "slice"
    
    print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

    dataloader = prepare_dataloader(total_iter * args.batch_size, args.batch_size, config.vocab_size)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    return model, config, dataloader, optimizer, criterion


def original(original_model: torch.nn.Module, config, dataloader, optimizer, criterion):
    model = deepcopy(original_model)
    model.cuda()
    model.mode = "original"

    tmp = validate(model_name, model, dataloader, criterion, 0, print_freq=1000, preserve_result=True)
    # model.cpu()
    del model

    return tmp 


def slice(original_model, config, dataloader, optimizer, criterion):
    model = deepcopy(original_model)
    model.mode = "slice"
    
    mslices : List[nn.Module] = []
    for i, layer_module in enumerate(model.bert.encoder.layer):
        mslices.append(layer_module)
    model.bert.encoder = offload_model(
        model=mslices, # 原生模型
        device=torch.device("cuda"), # 用于计算向前和向后传播的设备
        offload_device=torch.device("cpu"), # 模型将存储在其上的offload 设备
        num_slices=10, # 模型应分片的片数
        checkpoint_activation=False,
        # checkpoint_activation=True,
        num_microbatches=1
    )
    model.bert.to_cuda()  # only load embeddings to cuda
    model.cls.cuda()  # load final lm head to cuda

    tmp = validate(model_name, model, dataloader, criterion, 0, print_freq=1000, preserve_result=True)
    del model.bert
    del model.cls

    return tmp

def select(original_model, config, dataloader, optimizer, criterion):
    model = deepcopy(original_model)
    model.mode = "slice"
    
    mslices : List[nn.Module] = []
    for i, layer_module in enumerate(model.bert.encoder.layer):
        mslices.append(layer_module)
    model.bert.encoder = offload_model(  # 改select_model！
        model=mslices, # 原生模型
        device=torch.device("cuda"), # 用于计算向前和向后传播的设备
        offload_device=torch.device("cpu"), # 模型将存储在其上的offload 设备
        num_slices=10, # 模型应分片的片数
        checkpoint_activation=False,
        num_microbatches=1,
        device_list=[1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0]  # 1是在GPU内，0是不在
    )
    model.bert.to_cuda()  # only load embeddings to cuda
    model.cls.cuda()  # load final lm head to cuda

    tmp = validate(model_name, model, dataloader, criterion, 0, print_freq=1000, preserve_result=True)
    del model.bert
    del model.cls

    return tmp


def compare_result(result_a: List, result_b: List):
    if len(result_a) != len(result_b):
        print("The length of two results are different.")
        return False
    ret = True
    for i in range(len(result_a)):
        if not torch.allclose(result_a[i]['logits'], result_b[i]['logits']):
            print("The {}th result is different.".format(i))
            print(result_a[i]['logits'], result_b[i]['logits'])  # 在深度学习中，logits就是最终的全连接层的输出
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
    torch.cuda.empty_cache()

    time1 = time.time()
    seed_all(1)
    original_result = original(original_model, config, dataloader, optimizer, criterion)
    time2 = time.time()
    time_all = time2 - time1
    print ('The total time cost is: {}s'.format(time_all))
    torch.cuda.empty_cache()

    # compare
    print(compare_result(slice_result, original_result))
    del slice_result

    time1 = time.time()
    seed_all(1)
    select_result = select(original_model, config, dataloader, optimizer, criterion)
    time2 = time.time()
    time_all = time2 - time1
    print ('The total time cost is: {}s'.format(time_all))
    torch.cuda.empty_cache()

    # compare
    # print(compare_result(slice_result, original_result))
    print(compare_result(select_result, original_result))

