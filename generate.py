"""
    gpt2 benchmark 叶博代码+profiling
    能跑通，能打印
    TODO: dropout层的参数量为0
"""
import time
import torch
import torch.nn as nn
from dummy_dataloader import prepare_dataloader
from lib.transformers import GPT2Config, GPT2Model
from lib.transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import SGD, Adam
from train import train
from utils import get_parser, seed_all
from validate_old import validate
from pickle import dump

model_name = "gpt2"

def main():
    device_id = 1
    seed_all(1)
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model = GPT2LMHeadModel.from_pretrained("gpt2",pad_token_id=tokenizer.eos_token_id)  # 200这里原来是tokenizer.eos_token_id

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.cuda(device_id)

    dataloader = prepare_dataloader(2, 1, 50257)  # 原来是4

    # tell CUDA to start recording memory allocations
    torch.cuda.memory._record_memory_history(enabled='all')

    # input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt').to("cuda")
    for i, (images, target) in enumerate(dataloader):
        if device_id is not None:
            images = images.cuda(device_id, non_blocking=True)
            target = target.cuda(device_id, non_blocking=True)

        # compute output
        output = model.generate(images, max_length=100)

        pass


    # save a snapshot of the memory allocations
    s = torch.cuda.memory._snapshot()
    with open(f"snapshot.pickle", "wb") as f:
        dump(s, f)

    # tell CUDA to stop recording memory allocations now
    torch.cuda.memory._record_memory_history(enabled=None)

    # greedy_output = model.generate(input_ids, max_length=50)
    # print("Output:\n" + 100 * '-')
    # print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))



if __name__ == "__main__":
    time1 = time.time()
    seed_all(1)
    main()
    time2 = time.time()
    time_all = time2 - time1
    print ('The total time cost is: {}s'.format(time_all))
