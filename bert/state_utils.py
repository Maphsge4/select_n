import os
import shutil
from contextlib import contextmanager

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)


class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, arch, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 0
        self.arch = arch
        self.model = model
        self.optimizer = optimizer

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::

        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)


def load_checkpoint(
        checkpoint_file: str,
        device_id: int,
        arch: str,
        model: DistributedDataParallel,
        optimizer,  # SGD
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.

    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    state = State(arch, model, optimizer)

    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        print(f"=> loaded checkpoint file: {checkpoint_file}")

    # logic below is unnecessary when the checkpoint is visible on all nodes!
    # create a temporary cpu pg to broadcast most up-to-date checkpoint
    # with tmp_process_group(backend="gloo") as pg:
    #     rank = dist.get_rank(group=pg)

    #     # get rank that has the largest state.epoch
    #     epochs = torch.zeros(dist.get_world_size(), dtype=torch.int32)
    #     epochs[rank] = state.epoch
    #     dist.all_reduce(epochs, op=dist.ReduceOp.SUM, group=pg)
    #     t_max_epoch, t_max_rank = torch.max(epochs, dim=0)
    #     max_epoch = t_max_epoch.item()
    #     max_rank = t_max_rank.item()

    #     # max_epoch == -1 means no one has checkpointed return base state
    #     if max_epoch == -1:
    #         print(f"=> no workers have checkpoints, starting from epoch 0")
    #         return state

    #     # broadcast the state from max_rank (which has the most up-to-date state)
    #     # pickle the snapshot, convert it into a byte-blob tensor
    #     # then broadcast it, unpickle it and apply the snapshot
    #     print(f"=> using checkpoint from rank: {max_rank}, max_epoch: {max_epoch}")

    #     with io.BytesIO() as f:
    #         torch.save(state.capture_snapshot(), f)
    #         raw_blob = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8)

    #     blob_len = torch.tensor(len(raw_blob))
    #     dist.broadcast(blob_len, src=max_rank, group=pg)
    #     print(f"=> checkpoint broadcast size is: {blob_len}")

    #     if rank != max_rank:
    #         blob = torch.zeros(blob_len.item(), dtype=torch.uint8)
    #     else:
    #         blob = torch.as_tensor(raw_blob, dtype=torch.uint8)

    #     dist.broadcast(blob, src=max_rank, group=pg)
    #     print(f"=> done broadcasting checkpoint")

    #     if rank != max_rank:
    #         with io.BytesIO(blob.numpy()) as f:
    #             snapshot = torch.load(f)
    #         state.apply_snapshot(snapshot, device_id)

    #     # wait till everyone has loaded the checkpoint
    #     dist.barrier(group=pg)

    print(f"=> done restoring from previous checkpoint")
    return state


def save_checkpoint(state: State, is_best: bool, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:
        best = os.path.join(checkpoint_dir, "model_best.pth.tar")
        print(f"=> best model found at epoch {state.epoch} saving to {best}")
        shutil.copyfile(filename, best)

