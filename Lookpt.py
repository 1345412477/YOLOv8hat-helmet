import torch
model=torch.load('yolov8s.pt')
print(model)
with open('result.txt', 'w') as f:
    f.write(str(model))