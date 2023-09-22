import time
import torch
from metrics import AverageMeter, ProgressMeter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def train(
        train_loader: DataLoader,
        model: DistributedDataParallel,
        criterion,  # nn.CrossEntropyLoss
        optimizer,  # SGD,
        epoches: int,
        device_id: int,
        print_freq: int,
        batch_size: int,
        state=None
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    # target = torch.LongTensor(batch_size).random_(1000).cuda()

    prof = FlopsProfiler(model)  # add profiler
    # prof_step = len(train_loader) // 3  # 整除3，所以会在33%的时候输出profile！
    prof_step = 10  # debug
    
    start_epoch = 0 if state==None else state.epoch + 1
    for epoch in range(epoches):
        if state != None:   state.epoch = epoch
        for i, (images, target, batch_idx) in enumerate(train_loader):
            if batch_idx == prof_step:  # add profile
                prof.start_profile()
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.cuda(device_id, non_blocking=True)
            target = target.cuda(device_id, non_blocking=True)

            # compute output
            output = model(images).logits  # output: [batch_size, seq_len, vocab_size]
            # target: [batch_size, seq_len*0.15]

            if batch_idx == prof_step:  # add profile
                prof.print_model_profile(profile_step=batch_idx)
                prof.end_profile()

            loss = criterion(output[:, :int(512 * 0.15), :].permute((0, 2, 1)), target)

            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)
