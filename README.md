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

All images are labeled and preprocessed for training.  

## Results  
- The model achieves **high accuracy** in detecting screen-captured faces.  
- It can be extended to detect other spoofing techniques.  

## Future Development  
- Improve mAP by collecting more diverse data.  
- Optimize the model for mobile and real-time applications.  
- Integrate into online identity verification systems.  
