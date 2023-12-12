import torch
from torchvision import models
from pickle import dump

model = models.vit_l_16(weights='DEFAULT').cuda()
optimizer = torch.optim.Adam(model.parameters())

IMAGE_SIZE = 224

def train(model, optimizer):
  # create our fake image input: tensor shape is batch_size, channels, height, width
  fake_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()

  # call our forward and backward
  loss = model.forward(fake_image)
  loss.sum().backward()

  # optimizer update
  optimizer.step()
  optimizer.zero_grad()

# tell CUDA to start recording memory allocations
torch.cuda.memory._record_memory_history(enabled='all')

# train 3 steps
for _ in range(3):
  train(model, optimizer)

# save a snapshot of the memory allocations
s = torch.cuda.memory._snapshot()
with open(f"snapshot.pickle", "wb") as f:
    dump(s, f)

# tell CUDA to stop recording memory allocations now
torch.cuda.memory._record_memory_history(enabled=None)