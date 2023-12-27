import os

import torch
import argparse



def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Elastic ImageNet Training")
    # parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--data", default="./"
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        # choices=model_names,
        # help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        metavar="N",
        help="mini-batch size (default: 32), per worker (GPU)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=1000,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    parser.add_argument(
        "-m",
        "--mode",
        default="original",
    )

    parser.add_argument(
        "--dist-backend",
        default="nccl",
        choices=["nccl", "gloo"],
        type=str,
        help="distributed backend",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="/tmp/checkpoint.pth.tar",
        type=str,
        help="checkpoint file path, to load and save to",
    )
    parser.add_argument(
        "--iterations",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--training-or-inference",
        default="inference",
        choices=["training", "inference"],
        type=str
    )
    return parser


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)  # 这么设置使用确定性算法，如果代码中有算法cuda没有确定性实现，则会报错，可以验证代码中有没有cuda没有确定性实现的代码
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    
def adjust_learning_rate(optimizer, epoch, lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    learning_rate = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(1, -1).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class DummyProfile:
    """
    Dummy Profile.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        pass

    def step(self):
        pass
