import torch
import torchvision
import time
from torch.utils.data import DataLoader
import cv2
import os
from utils import MyDataset, get_transform, get_model, train_per_epoch, compute_map, plot_img

path_train = './RealFace Detector.v6i.tensorflow/train'
path_valid = './RealFace Detector.v6i.tensorflow/valid'
print(os.path.join(path_train, '_annotations.csv'))
train_set = MyDataset(path_train, transform= get_transform((224,224)))
val_set = MyDataset(path_valid, transform= get_transform((224,224)))

image, target, size = train_set.__getitem__(40)
plot_img(image,size[0], target = target)
# plt.show()
train_loader = DataLoader(train_set, batch_size=3, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))
val_loader = DataLoader(val_set, batch_size=3, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
best_mAP = 0

epochs = 15
num = 3
model = get_model(num)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr = 0.001, momentum= 0.9, weight_decay=0.0005)
# for epoch in range(epochs):
#     start_time = time.time()
#     train_loss = train_per_epoch(model, optimizer, train_loader, device)
#     mAP = compute_map(model, device, [1,2], val_loader)
#     if mAP > best_mAP:
#         best_mAP = mAP
#         torch.save(model.state_dict(), './model.pth')
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f'time: {elapsed_time // 60:.0f}m:{elapsed_time % 60:.0f}s')
#     print(f'Epoch: {epoch + 1}, loss: {train_loss:.4f}, mAP: {mAP:.4f}')
#     if best_mAP > 0.6:
#         break
#
# print('Best mAP:', best_mAP)
# print(next(model.parameters()).device)
# print(torch.cuda.is_available())  # Phải trả về True
# print(torch.cuda.device_count())  # Kiểm tra số GPU
# print(torch.cuda.get_device_name(0))