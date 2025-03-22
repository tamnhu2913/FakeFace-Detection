import cv2
import os
import sys
from utils import get_transform_test, get_model, plot_img, plot_img_cv2
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = get_model(3)
model.load_state_dict((torch.load('model1.pth', weights_only=True)))
model.to(device)
model.eval()

####image
# path = './new'
# nameImage = [file for file in os.listdir(path) if  file.lower().endswith('.jpg')]
# for name in nameImage:
#     image = cv2.imread(os.path.join(path, name))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     oriSize = image.shape[:-1]
#     transfromed = get_transform_test((500, 500))(image = image)
#     image = transfromed['image'].unsqueeze(0).to('cuda:0')
#     with torch.no_grad():
#         predict = model(image)
#     out_image = plot_img_cv2(image[0], size = oriSize, pred = predict[0])
#     while True:
#         out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
#         out_image = cv2.resize(out_image, (0,0), fx = 0.4, fy = 0.4)
#         cv2.imshow('Detection', out_image)
#         oriSize = [int(x * 0.5) for x in oriSize]
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             break
#         elif cv2.waitKey(0) & 0xFF == ord('k'):
#             cv2.destroyAllWindows()
#             sys.exit()
#         cv2.destroyWindow("Detection")

####video
# cap = cv2.VideoCapture('./new/video2.mp4')
#
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# size = (width, height)
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('./new/output.mp4', fourcc, fps, size)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print('video đã phát xong, thoát chương trình, ...')
#         break
#     # frame = cv2.resize(frame, (0,0), fx = 0.5, fy= 0.5)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     oriSize = frame.shape[:2]
#     transformed = get_transform_test((500,500))(image = frame)
#     frame = transformed['image'].unsqueeze(0).to('cuda:0')
#     with torch.no_grad():
#         predict = model(frame)
#     out_frame = plot_img_cv2(frame[0], thresh_score = 0.5, size=oriSize, pred=predict[0])
#     out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
#     # cv2.imshow('video', out_frame)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break
#     out.write(out_frame)
# cap.release()
# out.release()
# cv2.destroyAllWindows()

####webcame
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Không thể mở webcam! Kiểm tra camera hoặc quyền truy cập.")
    exit()

while True:
    ret, frame = cap.read()
    if ret:
        # frame = cv2.resize(frame, (0,0), fx = 0.5, fy= 0.5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        oriSize = frame.shape[:2]
        transformed = get_transform_test((500,500))(image = frame)
        frame = transformed['image'].unsqueeze(0).to('cuda:0')
        with torch.no_grad():
            predict = model(frame)
        out_frame = plot_img_cv2(frame[0], thresh_score = 0.7, size=oriSize, pred=predict[0])
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('video', out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()