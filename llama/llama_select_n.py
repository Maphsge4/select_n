from typing import List

import time
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from dummy_dataloader import prepare_dataloader
from lib.select_n import OffloadModel
from lib.transformers import LlamaConfig, LlamaForCausalLM
from utils import seed_all, get_parser
from validate_old import validate
from train import train

model_name = 'llama'

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    seed_all(1)
    device_id = 0
    torch.cuda.set_device(device_id)

    config = LlamaConfig.from_json_file("./llama/llama_7b_config.json")
    model =  LlamaForCausalLM(config=config)
    model.model.mode = "select"
    print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

    dataloader = prepare_dataloader(2 * args.batch_size, args.batch_size, config.vocab_size)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    mslices : List[nn.Module] = []
    for i, layer_module in enumerate(model.model.layers):
        mslices.append(layer_module)

    model.model.layers = OffloadModel(
        model=mslices,
        device=torch.device("cuda"),  # computation device
        offload_device=torch.device("cpu"),  # offload device
        num_slices=10, # currently not used
        checkpoint_activation=False,
        num_microbatches=1,
        device_list=[1, 0] * 16
    )
    model.to_cuda()

    print("现在的")
    print("max:", torch.cuda.max_memory_allocated(device=torch.device("cuda")))  # 显存量
    print("now", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量
                    
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