import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import cv2
import numpy as np

# Load the pre-trained model and set to evaluation mode
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

video_paths = [
    '/Users/sandeebadhikari/Documents/cs370-assignments/Project-1/YouTube-Videos/Cyclist and vehicle Tracking - 1.mp4',
    '/Users/sandeebadhikari/Documents/cs370-assignments/Project-1/YouTube-Videos/Cyclist and vehicle tracking - 2.mp4',
    '/Users/sandeebadhikari/Documents/cs370-assignments/Project-1/YouTube-Videos/Drone Tracking Video.mp4'
]

def detect_objects_in_frame(frame, model):
    # Convert the color space from BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    transform = T.Compose([T.ToTensor()])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    output = model(image)
    
    # Draw detected objects with a high score
    for idx, score in enumerate(output[0]['scores']):
        if score > 0.1:  # Confidence threshold
            label = output[0]['labels'][idx].item()
            if label in [2, 3]:  # COCO IDs: 2 for bicycle, 3 for car
                box = output[0]['boxes'][idx].tolist()
                # Convert box coordinates to integers
                box = list(map(int, box))
                # Draw bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label_text = f"{'Bicycle' if label == 2 else 'Car'}: {score:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

for video_path in video_paths:
    capture = cv2.VideoCapture(video_path)
    frame_rate = 5  # Extract frame every 30 seconds based on actual FPS
    
    frame_number = 0
    
    while True:
        success, frame = capture.read()
        if not success:
            break
        
        frame_number += 1
        
        if frame_number % frame_rate == 0:
            frame_with_detections = detect_objects_in_frame(frame, model)
            # Display the frame with detections
            cv2.imshow("Frame", frame_with_detections)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_number += 1     
    capture.release()
cv2.destroyAllWindows()
