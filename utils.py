import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import albumentations as A
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_transform(size):
  """
      Define a set of transformations for training images, including resizing, horizontal flipping,
      and converting to tensor format.
      """
  height, width = size
  transform = A.Compose([
    A.Resize(height = height, width = width),
    A.HorizontalFlip(p=0.3),
    A.ToFloat(max_value=255.0),
    A.pytorch.ToTensorV2()],
    bbox_params= A.BboxParams(format = 'pascal_voc', label_fields=['labels'])
    )
  return transform

def get_transform_test(size):
  """
  Define transformations for test images, including resizing and converting to tensor format.
  """
  height, width = size
  transform = A.Compose([
    A.Resize(height = height, width = width),
    A.ToFloat(max_value=255.0),
    A.pytorch.ToTensorV2()])
  return transform

class MyDataset(Dataset):
  """
  Custom dataset class for loading images and their corresponding bounding box annotations.
  """
  def __init__(self, path, transform = None):
    self.path = path
    self.transform = transform
    self.dataframe = pd.read_csv(os.path.join(path, '_annotations.csv'))
    self.filename = self.dataframe['filename'].unique()
    self.classes = {'real': 1, 'fake': 2}

  def __len__(self):
    """ Return the number of unique images in the dataset. """
    return len(self.filename)

  def __str__(self):
    """ Return a string representation of the dataset. """
    return f"Number of image: {len(self.filename)}, Total of bounding box: {len(self.dataframe)}"

  def __getitem__(self, index):
    """
    Load an image and its corresponding bounding boxes and labels.
    Apply transformations if specified.
    """
    name_image = self.filename[index]
    image_path = os.path.join(self.path, name_image)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    idx = self.dataframe['filename'] == name_image
    size = self.dataframe[idx][['height', 'width']].drop_duplicates().values
    bndbox = self.dataframe[idx][['xmin', 'ymin', 'xmax', 'ymax']].values
    label = self.dataframe[idx][['class']].values
    label = [self.classes[l[0]] for l in label]
    if self.transform:
      transformed = self.transform(image = image, bboxes = bndbox, labels = label)
      image = transformed['image']
      bndbox = transformed['bboxes']
      label = transformed['labels']
    target = {
        'boxes': torch.tensor(bndbox, dtype = torch.float32),
        'labels': torch.tensor(label, dtype = torch.int64)
    }
    return image, target, size

def convert_image(image, size):
  """
  Convert a PyTorch tensor image to a NumPy image and resize it to the given size.
  """
  H, W = size
  image = image.permute(1, 2, 0).cpu().numpy()
  image = cv2.resize(image, (W,H))
  image = (image * 255.0).astype(np.uint8)
  return image

def plot_box_label(image, bounding_boxes, labels, colors = ((0, 255, 0), (255, 0, 0))):
  """
  Draw bounding boxes and labels on an image using OpenCV.
  """
  for box, label in zip(bounding_boxes, labels):
    lab = 'real' if label == 1 else 'fake'
    color = colors[0] if label == 1 else colors[1]
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 8)
    x = np.max([5, box[0] - 10])
    y = np.max([0, box[1] - 5])
    cv2.putText(image, str(lab), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

def plot_img_cv2(image, size, thresh_score = 0.7, target = None, pred = None):
  """
  Visualize image with ground truth and predicted bounding boxes.
  """
  _, new_H, new_W = image.shape
  image = convert_image(image, size)
  scale_x = size[1] / new_W
  scale_y = size[0] / new_H
  if target:
    bndbox = target['boxes'].numpy() * [scale_x, scale_y, scale_x, scale_y]
    label = target['labels'].numpy()
    plot_box_label(image, bndbox, label)
  if pred:
    idx = torch.where(pred['scores']>thresh_score)[0]
    bndbox = pred['boxes'][idx].cpu().numpy()  * [scale_x, scale_y, scale_x, scale_y]
    label = pred['labels'][idx].cpu().numpy()
    plot_box_label(image, bndbox, label, colors=((0, 255, 255), (255, 255, 0)))
  return image

def get_model(num_classes):
  """
      Load a Faster R-CNN model with a ResNet-50 backbone and modify the classifier for custom classes.
  """
  model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model

def train_per_epoch(model, optimizer, data_loader, device):
  """
  Train the model for one epoch and compute the average loss.
  """
  model.train()
  train_loss = []
  # i = 0
  for images, targets, _ in data_loader:
    images = [image.to(device) for image in images]
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    # Forward step: Tính loss từ mô hình
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    # Backward step: Cập nhật gradient và tối ưu hóa
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    # print(f'Batch: {i}, \tloss of the batch: {losses.item():.4f}')
    # i += 1
    train_loss.append(losses.item())

    if torch.isnan(losses):
      print('Loss is nan')
      print('loss dict:', loss_dict)
      print('Targets:', targets)
      exit()

  train_loss = np.mean(train_loss)
  return train_loss

def mean_average_precision(detection_info):
    """
    input:
      detection_info (dictionary): key is number of class,
      [number of true boxes, score of predicted boxes, iou of predicted boxes]
    output:
      mAP: mean average precision
    """
    AP = []
    for clss in detection_info.keys():
      IOU = detection_info[clss][2]  # IOU của các bounding box dự đoán
      device = IOU.device  # Lấy device từ IOU

      # Chuyển detection_info[clss][0] (số box thật) thành tensor trên device
      num_true_boxes = detection_info[clss][0].float().to(device)

      TP = (IOU > 0.5).int()  # True positive
      FP = 1 - TP  # False positive

      csTP = torch.cumsum(TP, dim=0)  # Cumulative sum of True positive
      csFP = torch.cumsum(FP, dim=0)  # Cumulative sum of False positive

      recalls = csTP / num_true_boxes  # Đảm bảo num_true_boxes là tensor trên cùng device
      precisions = csTP / (csTP + csFP)

      # Chuyển tensor mặc định ở CPU sang `device`
      recalls = torch.cat((torch.tensor([0.0], device=device), recalls))
      precisions = torch.cat((torch.tensor([1.0], device=device), precisions))

      AP.append(torch.trapz(precisions, recalls))

    # print(sum(AP), len(AP))

    mAP = sum(AP) / len(AP) if len(AP) > 0 else torch.tensor(0.0, device=device)  # Tránh lỗi chia cho 0
    return mAP


def compute_map(model, device, classes, dataloader, threshold_scores=0.5):
  """
  Compute mean Average Precision (mAP) on the validation set or return predictions if it's a test set.

  Args:
      model (torch.nn.Module): The Faster R-CNN model or a similar object detection model.
      dataloader (DataLoader): DataLoader containing validation or test data.
      classes (list): List of class labels to compute mAP for.
      device (torch.device): Device to run the model on ('cuda' or 'cpu').
      is_test (bool): If True, returns model predictions instead of computing mAP.

  Returns:
      float: mAP if not a test set, or a list of model predictions if it's a test set.
  """
  detection_infor = {key: [torch.tensor(0, device=device),
                           torch.tensor([], device=device),
                           torch.tensor([], device=device)] for key in classes}
  model.eval()
  with torch.no_grad():
    for batch in dataloader:
      images, targets, _ = batch
      images = [image.to(device) for image in images]
      output = model(images)

      for true, pred in zip(targets, output):
        for i in classes:
          idx1 = torch.where(true['labels'] == i)[0]
          idx2 = torch.where((pred['labels'] == i) & (pred['scores'] > threshold_scores))[0]
          detection_infor[i][0] += len(idx1)  # Sửa chỗ này để tính đúng số lượng bbox GT

          if len(idx2) > 0:
            if len(idx1) == 0:
              iou = torch.zeros(len(idx2), device=device)
            else:
              iou, _ = torch.max(
                torchvision.ops.box_iou(pred['boxes'][idx2].to(device), true['boxes'][idx1].to(device)), axis=1)
              iou = iou.to(device)  # Đảm bảo iou nằm trên device

            detection_infor[i][1] = torch.cat((detection_infor[i][1], pred['scores'][idx2].to(device)))
            detection_infor[i][2] = torch.cat((detection_infor[i][2], iou))
  return mean_average_precision(detection_infor)
