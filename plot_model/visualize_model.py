import torch

model = torch.load(r"C:\Users\luzzi\Downloads\classes\cs251\car2ground_r0.5\weights\best.pt")

for k, v in model.items():
    print(f"{k}: {v}")
