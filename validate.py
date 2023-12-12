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
from utils import DummyProfile


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

    # switch to evaluate mode
    model.eval()

    prof = FlopsProfiler(model)  # add profiler
    # prof_step = len(val_loader) // 3  # 整除3，所以会在33%的时候输出profile！
    prof_step = 50  # debug

    profiler = DummyProfile() if not profiling else torch.profiler.profile(record_shapes=True, profile_memory=True, with_stack=True)
    results = []

    with torch.no_grad():
        end = time.time()
        gc.collect()
        with profiler as p:
            for i, (images, target) in enumerate(val_loader):
                if i == prof_step and profiling:  # add profile
                    prof.start_profile()
                if device_id is not None:
                    images = images.cuda(device_id, non_blocking=True)
                    target = target.cuda(device_id, non_blocking=True)

                # compute output
                output = model(images)
                if model_name == "gpt2":
                    output = output.last_hidden_state
                # del output
                # print(model)
                # loss = criterion(output[0], target)
                # print("loss: ", loss)  # debug

                if i == prof_step and profiling:  # add profile
                    prof.print_model_profile(profile_step=i)
                    prof.end_profile()

                # measure accuracy and record loss
                # acc1, acc5 = accuracy(output, target, topk=(1, 5))
                # losses.update(loss.item(), images.size(0))
                # top1.update(acc1[0], images.size(0))
                # top5.update(acc5[0], images.size(0))

                # measure elapsed time
                torch.cuda.synchronize(torch.cuda.current_device())
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)

                max_allocated, allocated = torch.cuda.max_memory_allocated(device=torch.device("cuda")), torch.cuda.memory_allocated(device=torch.device("cuda"))
                print(f"batch_time: {batch_time.val:.3f} ({batch_time.avg_without_first:.3f}), max_allocated: {max_allocated}, allocated: {allocated}")
                if preserve_result:
                    results.append(output)

            gc.collect()
        # p.export_memory_timeline(str(trace_dir.joinpath(f"linear_stack_{now}.html")), torch.cuda.current_device())


        # TODO: this should also be done with the ProgressMeter
        # print(
        #     " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        # )

    return results if preserve_result else None

