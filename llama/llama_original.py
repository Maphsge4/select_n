"""
    gpt2 benchmark 叶博代码+profiling
    能跑通，能打印
    TODO: dropout层的参数量为0
"""
import time
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from dummy_dataloader import prepare_dataloader
from lib.transformers import LlamaConfig, LlamaForCausalLM
from utils import seed_all, get_parser
from validate import validate
from train import train

model_name = 'llama'

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    seed_all(1)
    device_id = 0
    torch.cuda.set_device(device_id)

    config = LlamaConfig.from_json_file("./llama/llama_7b_config.json")
    model = LlamaForCausalLM(config=config)
    model.cuda()
    print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

    dataloader = prepare_dataloader(4 * args.batch_size, args.batch_size, config.vocab_size)
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