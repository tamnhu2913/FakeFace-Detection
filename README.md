# FakeFace-Detection: Identifying Screen-Captured Faces  

## Project Description  
**FakeFace-Detection** focuses on detecting whether a face is **screen-captured** from a video, ID card, or mobile device. This project helps identify fraud attempts in online identity verification systems.  

## Objectives  
- Distinguish real faces from screen-captured faces.  
- Apply the model in eKYC, account security, and fraud prevention.  

## Model Used  
- **Model:** Faster R-CNN  
- **mAP:** > 0.65  
- **Loss:** ~0.11  

## Dataset  
The dataset includes:  
- **Real faces** (directly captured images).  
- **Fake faces** (captured from screens, videos, or ID cards).

| **Real Example** | **Fake Example** |
|------------------|------------------|
| <img src='https://github.com/user-attachments/assets/54641156-b87a-4007-9a59-f92d98edf17a' width="400" height="321"> | <img src='https://github.com/user-attachments/assets/8e2a0f6e-60cb-489a-9424-434f327c8b57' width="400" height="321"> |

All images are labeled and preprocessed for training.  

## Results  
- The model achieves **high accuracy** in detecting screen-captured faces.  
- It can be extended to detect other spoofing techniques.  

https://github.com/user-attachments/assets/cc967c05-0afd-4877-b6f0-57cf18abdf9e

https://github.com/user-attachments/assets/f4db5601-e635-4ec6-856b-6fbbdb3ab255

## Future Development  
- Improve mAP by collecting more diverse data.  
- Optimize the model for mobile and real-time applications.  
- Integrate into online identity verification systems.  
