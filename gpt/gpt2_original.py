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
from torch.optim import SGD, Adam
from train import train
from utils import get_parser, seed_all
from validate_old import validate

model_name = "gpt2"

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    seed_all(1)
    device_id = 0
    torch.cuda.set_device(device_id)

    config = GPT2Config.from_json_file("./gpt/gpt2_large_config.json")
    model = GPT2Model(config=config)
    model.mode = "original"
    model.cuda()
    print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

    # dataloader = prepare_dataloader(2 * args.batch_size, args.batch_size, config.vocab_size)  # 原来是4
    dataloader = prepare_dataloader(2, 1, 50257)  # 原来是4
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
