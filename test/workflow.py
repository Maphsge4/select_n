from timer import Timer
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
import math
import torch.nn.init as init
import torch.optim as optim
import gpustat
import os
import logging
from copy import deepcopy
from gpu_memory import *
import torch.utils.checkpoint

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def make_deterministic(seed):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float64) if check_correctness else None


def check_basic():
    hostnames = os.getenv('SLURM_JOB_NODELIST')
    if os.getenv('CUDA_VISIBLE_DEVICES') and ',' in os.getenv('CUDA_VISIBLE_DEVICES'):
        gpu_id = int(os.getenv('CUDA_VISIBLE_DEVICES').split(',')[0])
    else:
        gpu_id = int(os.getenv('CUDA_VISIBLE_DEVICES', 0))
    print(f"hostname: {hostnames}, gpu_id: {gpu_id}")
    gpu = gpustat.new_query().gpus[gpu_id]
    if len(gpu.processes) != 0:
        print(gpu)
        assert False


model_dim = 1536
hidden_size = 6144
activation_fn = torch.nn.functional.relu
num_experts = 10
gpu = torch.device(f'cuda:0')
cpu = torch.device(f'cpu')
global_timer = Timer()
total_iters = 3
check_correctness = False
fake_memory = False
enable_activation_checkpoint = True
enable_saved_on_cpu = False


def get_current_gpu_memory():
    if fake_memory:
        return 0
    # torch.cuda.synchronize()
    return torch.cuda.memory_allocated()
    import gpustat
    gpu_id = int(os.getenv('CUDA_VISIBLE_DEVICES', 0))
    gpu = gpustat.new_query().gpus[gpu_id]
    return gpu.memory_used


class LinearPinned(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    batched: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(LinearPinned, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.if_bias = bias
        # self._grads/self._grads_gpu; self._params/self._params_gpu
        self.init_buffers()
        self.update_wandb()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def update_wandb(self):
        # self.weight & self.bias should always point to the valid device
        self.weight = torch.nn.Parameter(self._params[:self.out_features,:-1])
        self.weight.grad = self._grads[:self.out_features,:-1]
        if self.if_bias:
            self.bias = torch.nn.Parameter(self._params[:,-1:].view(-1))
            self.bias.grad = self._grads[:,-1:].view(-1)
        else:
            self.register_parameter('bias', None)
        self._params.grad = self._grads

    def init_buffers(self):
        # currently always hold _params buffer on GPU to avoid zero grad problem
        self._params = torch.zeros((self.out_features, self.in_features + 1)).pin_memory()
        self._grads = torch.zeros((self.out_features, self.in_features + 1)).pin_memory()

    def to(self, device, non_blocking=False, with_grad=False):
        if device != cpu:
            if with_grad:
                # self._grads_gpu.storage().resize_(self._grads.storage().size())
                self._grads_gpu = self._grads.to(device, non_blocking=non_blocking)
                self.weight.grad.data = self._grads_gpu[:self.out_features,:-1]
                self.bias.grad.data = self._grads_gpu[:,-1:].view(-1)
            # create gpu buffers
            # self._params_gpu.storage().resize_(self._params.storage().size())
            self._params_gpu = self._params.to(device, non_blocking=non_blocking)
            # update params and grads to gpu buffer
            self.weight.data = self._params_gpu[:self.out_features,:-1]
            self.bias.data = self._params_gpu[:,-1:].view(-1)
            # self._params_gpu.grad = self._grads_gpu
        
        elif device == cpu:
            if with_grad:
                self._grads.data.copy_(self._grads_gpu, non_blocking=non_blocking)
                self.weight.grad.data = self._grads[:self.out_features,:-1]
                self.bias.grad.data = self._grads[:,-1:].view(-1)
                # self._grads_gpu.storage().resize_(0)
                del self._grads_gpu
            # copy gpu buffers to cpu
            self._params.data.copy_(self._params_gpu, non_blocking=non_blocking)
            # update params and grads to cpu buffer
            self.weight.data = self._params[:self.out_features,:-1]
            self.bias.data = self._params[:,-1:].view(-1)
            # self._params.grad = self._grads
            # self._params_gpu.storage().resize_(0)
            del self._params_gpu
            
        # assert self.weight.device == device, f"weight device {self.weight.device} != {device}"

    def release_gpu_memory(self):
        del self.batched
        del self.weight
        del self.bias
        self.batched = self.batched_cpu
        self.update_wandb()


class ExpertModelPinned(torch.nn.Module):
    def __init__(self, model_dim, hidden_size, activation_fn, to_gpu=False):
        super().__init__()
        self.activation_fn = activation_fn
        self.fc1 = LinearPinned(model_dim, hidden_size, bias=True)
        self.fc2 = LinearPinned(hidden_size, model_dim, bias=True)
        self.current_stream = torch.cuda.current_stream()
        if to_gpu:
            self.to(device=gpu, non_blocking=False)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x

    def to(self, device, non_blocking=True, with_grad=False):
        self.fc1.to(device=device, non_blocking=non_blocking, with_grad=with_grad)
        self.fc2.to(device=device, non_blocking=non_blocking, with_grad=with_grad)

    def is_cuda(self):
        # may encounter data inconsistency here; must sync before check is_cuda
        return hasattr(self.fc1, '_params_gpu') and hasattr(self.fc2, '_params_gpu')

    def where(self):
        print(self.fc1.batched.device, self.fc2.batched.device, self.fc1.batched.data_ptr(), self.fc2.batched.data_ptr())
        print(self.fc1.weight.device, self.fc2.weight.device, self.fc1.weight.data_ptr(), self.fc2.weight.data_ptr())
        print(self.fc1.bias.device, self.fc2.bias.device, self.fc1.bias.data_ptr(), self.fc2.bias.data_ptr())

experts = None

def is_all_zero(t: torch.Tensor):
    return torch.equal(t, torch.zeros(t.shape, device=t.device))


class ExpertLoading(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, expert_index, expert=None, expert_models=None):
        see_memory_usage("workflow fwd before loading", reset=False)
        nvtx.range_push(f"ExpertLoading {expert_index} forward")
        ctx.expert_index = expert_index
        ctx.expert = expert
        experts = expert_models
        assert expert_index < num_experts
        # should wait for last backward and its own prefetching
        if expert.current_stream is not None:
            expert.current_stream.synchronize()
        nvtx.range_push(f"expert {expert_index} loading")
        if not expert.is_cuda():
            logger.info(f"expert {expert_index} not on cuda, loading now")
            with torch.cuda.stream(expert.current_stream):
                expert.to(device=gpu, non_blocking=True)
            expert.current_stream.synchronize()
            assert expert.current_stream.query()
        nvtx.range_pop()
        last_expert = expert_index == num_experts - 1
        # check and prefetch next expert to GPU
        if not last_expert:
            expert_next = experts[expert_index + 1]
            nvtx.range_push(f"expert {expert_index + 1} prefetching")
            s1 = torch.cuda.Stream(device=gpu)
            expert_next.current_stream = s1
            with torch.cuda.stream(s1):
                logger.info(f"fwd: prefetching expert {expert_index + 1}")
                expert_next.to(device=gpu, non_blocking=True, with_grad=expert_index + 1 == num_experts - 1)
            nvtx.range_pop()
        nvtx.range_pop()
        see_memory_usage("workflow fwd after loading", reset=False)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        see_memory_usage("workflow bwd before offloading", reset=False)
        nvtx.range_push(f"ExpertLoading {ctx.expert_index} backward")
        expert_index = ctx.expert_index
        first_expert = expert_index == 0
        # current expert move to cpu
        expert = ctx.expert
        # print(f"expertLoading backward: {expert.fc1.weight.grad}")
        # should wait for the backward computation to finish
        s = torch.cuda.current_stream()
        s.synchronize()
        # sync grads using expert.current_stream
        logger.info(f"bwd: offloading expert {expert_index} grads")
        s1 = torch.cuda.Stream(device=gpu)
        with torch.cuda.stream(s1):
            expert.to(device=cpu, non_blocking=True, with_grad=True)
            expert.current_stream = s1
        # # if not first_expert, need to offload parameter
        # if not first_expert:
        #     logger.info(f"bwd: offloading expert {expert_index} params")
        #     s1 = torch.cuda.Stream(device=device)
        #     with torch.cuda.stream(s1):
        #         expert.to(device=torch.device('cpu'), non_blocking=True)
        #         expert.current_stream = s1
        nvtx.range_pop()
        see_memory_usage("workflow bwd after loading", reset=False)
        return grad_output, None, None, None


class ExpertOffLoading(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, expert_index, expert=None, expert_models=None):
        see_memory_usage("workflow fwd before offloading", reset=False)
        ctx.expert_index = expert_index
        ctx.expert = expert
        ctx.expert_models = expert_models
        nvtx.range_push(f"ExpertOffLoading {expert_index} forward")
        last_expert = expert_index == num_experts - 1
        # current expert move to cpu
        if not last_expert:
            nvtx.range_push(f"expert {expert_index} offloading")
            # should wait for the forward computation to finish
            # but should not block current stream
            s = torch.cuda.current_stream()
            s.synchronize()
            s1 = torch.cuda.Stream(device=gpu)
            with torch.cuda.stream(s1):
                logger.info(f"fwd: offloading expert {expert_index}")
                expert.to(device=cpu, non_blocking=True)
                nvtx.range_pop()
                ctx.current_s = s1
        else:
            ctx.current_s = None
        nvtx.range_pop()
        see_memory_usage("workflow fwd after offloading", reset=False)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        see_memory_usage("workflow bwd before loading", reset=False)
        expert_index = ctx.expert_index
        expert = ctx.expert
        experts = ctx.expert_models
        s = ctx.current_s
        # should ensures backward after forward
        # after here, current stream should be the same as ctx.current_s
        if s is not None:
            s.synchronize()
            assert s.query()
        # ensures forward computation is finished
        s = torch.cuda.current_stream()
        s.synchronize()

        # torch.cuda.current_stream().wait_stream(s)

        nvtx.range_push(f"ExpertOffLoading {expert_index} backward")
        assert expert_index < num_experts
        # ensure last prefetching has finished
        if expert.current_stream is not None:
            expert.current_stream.synchronize()
            assert expert.current_stream.query()
        if not expert.is_cuda():
            logger.info(f"expert {expert_index} not on cuda, loading now")
            # should make backward computation wait for the loading, dispatch on default stream, but not waiting
            expert.to(device=gpu, non_blocking=True, with_grad=True)
        # now, current expert should on GPU; create grad buffer and set param.grad to the buffer
        # check and prefetch next expert to GPU
        first_expert = expert_index == 0
        if not first_expert:
            expert_next = experts[expert_index - 1]
            s1 = torch.cuda.Stream(device=gpu)
            expert_next.current_stream = s1
            with torch.cuda.stream(s1):
                logger.info(f"bwd: prefetching expert {expert_index - 1}")
                expert_next.to(device=gpu, non_blocking=True, with_grad=True)
        nvtx.range_pop()
        return grad_output, None, None, None

class SaveOnCpu():
    def __init__(self, enter_result=None):
        self.pack_hook = torch.autograd.graph.save_on_cpu().pack_hook
        self.unpack_hook = torch.autograd.graph.save_on_cpu().unpack_hook
        self.enter_result = enter_result
    def __enter__(self):
        if not enable_saved_on_cpu:
            return self.enter_result
        torch._C._autograd._register_saved_tensors_default_hooks(self.pack_hook, self.unpack_hook)

    def __exit__(self, *args):
        if not enable_saved_on_cpu:
            return
        torch._C._autograd._reset_saved_tensors_default_hooks()


class DummyExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([ExpertModelPinned(model_dim, hidden_size, activation_fn) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        for i, expert in enumerate(self.experts):
            x = ExpertLoading.apply(x, i, expert, self.experts)
            nvtx.range_push(f"expert {i} ffn")
            with SaveOnCpu():
                x = torch.utils.checkpoint.checkpoint(expert, x) if enable_activation_checkpoint else expert(x)
            see_memory_usage("workflow fwd before ffn", reset=False)
            nvtx.range_pop()
            x = ExpertOffLoading.apply(x, i, expert, self.experts)
        return x
    
    def to(self, device):
        # only for debugging purpose
        assert False
        for expert in self.expert_models:
            expert.to(device)


class baselineLoading(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, expert_index, expert=None):
        ctx.expert_index = expert_index
        ctx.expert = expert
        a = get_current_gpu_memory()
        expert.to(device=gpu)
        logger.debug(f"fwd: baselineLoading: {get_current_gpu_memory() - a}")
        return x

    @staticmethod
    def backward(ctx, grad_output):
        expert_index = ctx.expert_index
        expert = ctx.expert
        a = get_current_gpu_memory()
        expert.to(device=torch.device('cpu'))
        logger.debug(f"bwd: baselineLoading: {get_current_gpu_memory() - a}")
        return grad_output, None, None

class baselineOffLoading(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, expert_index, expert=None):
        ctx.expert_index = expert_index
        ctx.expert = expert
        a = get_current_gpu_memory()
        expert.to(device=torch.device('cpu'))
        logger.debug(f"fwd: baselineOffLoading: {get_current_gpu_memory() - a}")
        return x

    @staticmethod
    def backward(ctx, grad_output):
        expert_index = ctx.expert_index
        expert = ctx.expert
        a = get_current_gpu_memory()
        expert.to(device=gpu)
        logger.debug(f"bwd: baselineOffLoading: {get_current_gpu_memory() - a}")
        return grad_output, None, None


class baselineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([ExpertModel(model_dim, hidden_size, activation_fn) for _ in range(num_experts)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, expert in enumerate(self.experts):
            x = baselineLoading.apply(x, i, expert)
            x = expert(x)
            x = baselineOffLoading.apply(x, i, expert)
        return x


class ExpertModel(torch.nn.Module):
    def __init__(self, model_dim, hidden_size, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        self.fc1 = torch.nn.Linear(model_dim, hidden_size, bias=True)
        self.fc2 = torch.nn.Linear(hidden_size, model_dim, bias=True)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x

def summary(iter_times):
    if len(iter_times) == 1:
        return iter_times[0]
    return (sum(iter_times) - max(iter_times)) / (len(iter_times) - 1)


def model_size(model: torch.nn.Module):
    size = 0
    for name, params in model.named_parameters():
        size += params.numel() * params.element_size()
    return size


def allclose(a, b):
    return torch.allclose(a, b)


class result():
    def __init__(self, name, x, losses, grads, old_params, weights, optimizer_state=None, iter_time=None):
        self.name = name
        self.x = x
        self.losses = losses
        self.grads = grads
        self.old_params = old_params
        self.weights = weights
        self.optimizer_state = optimizer_state
        self.iter_time = iter_time
    
    def __eq__(self, other) -> bool:
        if self is None or other is None:
            return True
        assert len(self.x) == len(other.x)
        for i in range(len(self.x)):
            assert allclose(self.x[i], other.x[i])

        ret = True

        for i in range(len(self.losses)):
            if not allclose(self.losses[i], other.losses[i]):
                logger.error(f"loss mismatch {i}: {self.losses[i]}, {other.losses[i]}")
                ret = False
                break

        for key in self.old_params.keys():
            for i in range(len(self.old_params[key])):
                if not allclose(self.old_params[key][i], other.old_params[key][i]):
                    logger.error(f"old params mismatch: {key} {i}, shape: {self.old_params[key][i].shape}")
                    logger.warning(f"old params: {self.old_params[key][i]} mismatches {other.old_params[key][i]}")
                    ret = False
                    break

        for key in self.grads.keys():
            for i in range(len(self.grads[key])):
                # assert not is_all_zero(self.grads[key][i])
                if not allclose(self.grads[key][i], other.grads[key][i]):
                    logger.error(f"grads mismatch: {key} {i}, shape: {self.grads[key][i].shape}")
                    logger.warning(f"grads: {self.grads[key][i]} mismatches {other.grads[key][i]}")
                    ret = False
                    break

        for key in self.optimizer_state.keys(): # step, square_avg, acc_delta
            for i in range(len(self.optimizer_state[key])): # 4 params * iters
                if not allclose(self.optimizer_state[key][i], other.optimizer_state[key][i]):
                    logger.error(f"optimizer_states mismatch: {key} {i}, shape: {self.optimizer_state[key][i].shape}")
                    logger.warning(f"optimizer_states: {self.optimizer_state[key][i]} mismatches {other.optimizer_state[key][i]}")
                    return False
        
        for key in self.weights.keys():
            for i in range(len(self.weights[key])):
                if not allclose(self.weights[key][i], other.weights[key][i]):
                    logger.error(f"params mismatch: {key} {i}, shape: {self.weights[key][i].shape}")
                    logger.warning(f"params: {self.weights[key][i]} mismatches {other.weights[key][i]}")
                    ret = False
                    break
        
        return ret
        

def baseline():
    print("baseline")
    model = baselineModel()
    logger.debug(f"model size: {model_size(model)}")
    iter_times = []
    losses = []
    from collections import defaultdict
    weights = defaultdict(list)
    grads = defaultdict(list)
    old_params = defaultdict(list)
    optimizer_state = defaultdict(list)
    oldx = []
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    for _ in range(total_iters):
        x = torch.randn(5344, model_dim, device=gpu)
        x.requires_grad_(True)
        if check_correctness:
            old_x = x.detach().cpu()
            oldx.append(old_x)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        global_timer.start()
        nvtx.range_push("baseline_forward")
        y = model(x)
        nvtx.range_pop()
        loss = y.mean()
        losses.append(loss.detach().cpu()) if check_correctness else None
        nvtx.range_push("baseline_backward")
        loss.backward()
        nvtx.range_pop()
        torch.cuda.synchronize()
        global_timer.stop()
        iter_time = global_timer.elapsed()
        global_timer.reset()
        iter_times.append(iter_time)
        if check_correctness:
            for name, param in model.named_parameters():
                old_params[name].append(deepcopy(param.detach().cpu()))
            for name, param in model.named_parameters():
                grads[name].append(deepcopy(param.grad.detach().cpu()))
        optimizer.step()
        if check_correctness:
            for param in model.parameters():
                param_state = optimizer.state[param]
                for k, v in param_state.items():
                    if isinstance(v, int):
                        continue
                    optimizer_state[k].append(deepcopy(v.detach().cpu()))
            for name, param in model.named_parameters():
                weights[name].append(deepcopy(param.detach().cpu()))
    print(f"baseline: {summary(iter_times)}")
    return result("baseline", oldx, losses, grads, old_params, weights, optimizer_state, summary(iter_times))

class naiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([ExpertModel(model_dim, hidden_size, activation_fn) for _ in range(num_experts)])
    
    def forward(self, x: torch.Tensor):
        for expert in self.experts:
            x = expert(x)
        return x

def no_copy():
    print("no_copy")
    model = naiveModel()
    model.to(device=gpu)
    torch.cuda.synchronize()
    iter_times = []
    losses = []
    from collections import defaultdict
    weights = defaultdict(list)
    grads = defaultdict(list)
    old_params = defaultdict(list)
    optimizer_state = defaultdict(list)
    oldx = []
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    for _ in range(total_iters):
        x = torch.randn(5344, model_dim, device=gpu)
        x.requires_grad_(True)
        if check_correctness:
            old_x = x.detach().cpu()
            oldx.append(old_x)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        nvtx.range_push("no_copy_forward")
        global_timer.start()
        see_memory_usage("no_copy before forward", reset=False)
        y = model(x)
        see_memory_usage("no_copy after forward", reset=False)
        nvtx.range_pop()
        loss = y.mean()
        losses.append(loss.detach().cpu()) if check_correctness else None
        nvtx.range_push("no_copy_backward")
        loss.backward()
        see_memory_usage("no_copy after backward", reset=False)
        nvtx.range_pop()
        torch.cuda.synchronize()
        global_timer.stop()
        iter_time = global_timer.elapsed()
        global_timer.reset()
        iter_times.append(iter_time)

        if check_correctness:
            for name, param in model.named_parameters():
                old_params[name].append(deepcopy(param.detach().cpu()))
            for name, param in model.named_parameters():
                grads[name].append(deepcopy(param.grad.detach().cpu()))

        optimizer.step()

        if check_correctness:
            for param in model.parameters():
                param_state = optimizer.state[param]
                for k, v in param_state.items():
                    if isinstance(v, int):
                        continue
                    optimizer_state[k].append(deepcopy(v.detach().cpu()))
            for name, param in model.named_parameters():
                weights[name].append(deepcopy(param.detach().cpu()))
        see_memory_usage("no_copy iter end", reset=False)
    print(f"no_copy: {summary(iter_times)}")
    return result("no_copy", oldx, losses, grads, old_params, weights, optimizer_state, summary(iter_times))


def workflow():
    print("workflow")
    model = DummyExpert()
    torch.cuda.synchronize()
    iter_times = []
    losses = []
    from collections import defaultdict
    weights = defaultdict(list)
    grads = defaultdict(list)
    old_params = defaultdict(list)
    optimizer_state = defaultdict(list)
    oldx = []
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    for _ in range(total_iters):
        x = torch.randn(5344, model_dim, device=gpu)
        x.requires_grad_(True)
        if check_correctness:
            old_x = x.detach().cpu()
            oldx.append(old_x)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        global_timer.start()
        # print(f"before forward: {get_current_gpu_memory()}")
        y = model(x)
        # print(f"after forward: {get_current_gpu_memory()}")
        gpu_memory_summary(model=model)
        loss = y.mean()
        losses.append(loss.detach().cpu()) if check_correctness else None
        # print(f"before backward: {get_current_gpu_memory()}")
        loss.backward()
        torch.cuda.synchronize()
        gpu_memory_summary(model=model)
        # print(f"after backward: {get_current_gpu_memory()}")
        global_timer.stop()
        iter_time = global_timer.elapsed()
        global_timer.reset()
        iter_times.append(iter_time)
        if check_correctness:
            for name, param in model.named_parameters():
                old_params[name].append(deepcopy(param.detach().cpu()))
            for name, param in model.named_parameters():
                grads[name].append(deepcopy(param.grad.detach().cpu()))
        optimizer.step()
        if check_correctness:
            for param in model.parameters():
                param_state = optimizer.state[param]
                for k, v in param_state.items():
                    if isinstance(v, int):
                        continue
                    optimizer_state[k].append(deepcopy(v.detach().cpu()))

            for name, param in model.named_parameters():
                weights[name].append(deepcopy(param.detach().cpu()))
    print(f"workflow: {summary(iter_times)}")
    return result("workflow", oldx, losses, grads, old_params, weights, optimizer_state, summary(iter_times))


def main():
    check_basic()
    baseline_result = default_result = workflow_result = None

    make_deterministic(233)
    nvtx.range_push('baseline')
    # baseline_result = baseline()
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    make_deterministic(233)
    nvtx.range_push('workflow')
    workflow_result = workflow()
    torch.cuda.synchronize()
    # only workflow max_memory_allocated: 864150528
    see_memory_usage("workflow", reset=True)
    nvtx.range_pop()

    make_deterministic(233)
    nvtx.range_push('no_copy')
    default_result = no_copy()
    # only default max_memory_allocated: 4974566912
    see_memory_usage("no_copy", reset=True)
    torch.cuda.synchronize()
    nvtx.range_pop()

    if check_correctness:
        assert baseline_result == default_result
        assert workflow_result == default_result
    

if __name__ == '__main__':
    main()
