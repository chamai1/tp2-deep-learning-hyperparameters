import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=1e-4
)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(20):
    optimizer.zero_grad()

    x = torch.randn(32, 128)
    y = torch.randint(0, 10, (32,))

    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()