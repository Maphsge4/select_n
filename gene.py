import datetime
import gc
import os
import pathlib
import time

import torch
from lib.fragile import fragile
from lib.profiler import FlopsProfiler
from metrics import AverageMeter, ProgressMeter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from pickle import dump


def validate(
        model_name,
        model: DistributedDataParallel,
        val_loader: DataLoader,
        criterion, 
        device_id: int,
        print_freq: int,
        profiling: bool = False,
        preserve_result: bool = False,
):
    trace_dir = pathlib.Path(__file__).parent.joinpath("traces")
    now = datetime.datetime.now().strftime("%Y_%m_%d:%H.%M.%S")
    trace_dir.mkdir(exist_ok=True)

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time], prefix="Test: "
    )

    # from lib.transformers import GPT2LMHeadModel, GPT2Tokenizer

    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model.cuda(device_id)

    # # tell CUDA to start recording memory allocations
    torch.cuda.memory._record_memory_history(enabled='all')
    use_cache = True

    for i, (images, target) in enumerate(val_loader):
        if device_id is not None:
            print(f"input shape: {images.shape}")
            images = images.cuda(device_id, non_blocking=True)
            target = target.cuda(device_id, non_blocking=True)

        # compute output
        output = model.generate(images, max_length=156, use_cache=use_cache)
        print(output)

        pass

    # save a snapshot of the memory allocations
    s = torch.cuda.memory._snapshot()
    with open(f"snapshot.pickle", "wb") as f:
        dump(s, f)

    # tell CUDA to stop recording memory allocations now
    torch.cuda.memory._record_memory_history(enabled=None)