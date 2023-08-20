import torch
import torch.nn as nn

if torch.backends.mps.is_available():
    print("Yes")

device = torch.device("cuda")
print(device)

m = nn.Linear(20, 30)
input = torch.randn(128, 8, 20)
output = m(input)
print(output.size())
