import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
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
    detections = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the color space from BGR (OpenCV) to RGB (PIL)
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
                detection = {'box': box, 'score': score, 'label': label}
                detections.append(detection)
                box = list(map(int, box))
                # Draw bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label_text = f"{'Bicycle' if label == 2 else 'Car'}: {score:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame, detections


def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state variables, 2 measurements (x and y positions)
    kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                     [0, 1, 0, 0]])
    kf.R *= np.array([[1, 0],       # Measurement noise
                      [0, 1]]) 
    kf.P *= 10.                     # Initial state covariance
    kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.01, block_size=2)  # Process noise
    return kf

kf = initialize_kalman_filter()

def update_kalman_filter(kf, detection):
    # Assume detection is a bounding box [x1, y1, x2, y2]
    x_center = (detection[0] + detection[2]) / 2
    y_center = (detection[1] + detection[3]) / 2
    
    # Update Kalman filter with detected position
    kf.update(np.array([x_center, y_center]))
    
    # Predict next state
    kf.predict()
    
    return kf

trackers = []

def match_detection_to_tracker(detection, trackers):
    min_distance = float('inf')
    matched_tracker_index = None
    
    detection_center = np.array([(detection['box'][0] + detection['box'][2]) / 2, 
                                 (detection['box'][1] + detection['box'][3]) / 2])
    
    for i, tracker in enumerate(trackers):
        predicted_position = tracker['predicted_position']
        distance = np.linalg.norm(detection_center - predicted_position)
        
        if distance < min_distance:
            min_distance = distance
            matched_tracker_index = i
    
    distance_threshold = 50  
    
    if min_distance < distance_threshold:
        return matched_tracker_index
    else:
        return None

for video_path in video_paths:
    capture = cv2.VideoCapture(video_path)
    frame_rate = 20  # Extract frame every 20 seconds 
    
    frame_number = 0
    
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_for_prediction = frame.copy()
        
        frame_number += 1 
        
        if frame_number % frame_rate == 0:
            frame_with_detections, detections = detect_objects_in_frame(frame, model)
    
            for detection in detections:
                matched_tracker_index = match_detection_to_tracker(detection, trackers)
                if matched_tracker_index is not None:
                    kf = trackers[matched_tracker_index]['kf']
                    update_kalman_filter(kf, detection['box'])
                    # Extract the updated position after prediction and update
                    x_position, y_position = int(kf.x[0, 0]), int(kf.x[1, 0])
                    trackers[matched_tracker_index]['predicted_position'] = np.array([x_position, y_position])
                else:
                    new_kf = initialize_kalman_filter()
                    # Assuming the detection's center as the initial position
                    x_center = (detection['box'][0] + detection['box'][2]) / 2
                    y_center = (detection['box'][1] + detection['box'][3]) / 2
                    new_kf.x = np.array([x_center, y_center, 0., 0.]).reshape(4, 1)  # Corrected shape for state vector
                    new_tracker = {
                        'kf': new_kf,
                        'predicted_position': np.array([x_center, y_center]),  # Initialize predicted_position
                        'positions': [(int(x_center), int(y_center))]  # Initialize positions list with the current position
                    }
                    trackers.append(new_tracker)

                
            # Correct way to access the 'kf' key of the first tracker in the list, as an example
            if trackers:  # Ensure the list is not empty
                kf = trackers[0]['kf']  # Accessing the first tracker and then its 'kf' key

            trajectory_frame = np.zeros_like(frame_for_prediction)
            
            for tracker in trackers:
                kf = tracker['kf']
                kf.predict()
                x_position, y_position = int(kf.x[0, 0]), int(kf.x[1, 0])
                #print("kf is defined:", 'kf' in locals() or 'kf' in globals())  # Check if kf is defined
                #print("Type of kf:", type(kf))  # Check the type of kf
    
                if 'positions' not in tracker:
                    tracker['positions'] = []
                tracker['positions'].append((x_position, y_position))
    
                for i in range(1, len(tracker['positions'])):
                    cv2.line(trajectory_frame, tracker['positions'][i - 1], tracker['positions'][i], (0, 255, 0), 2)
                
            combined_frame = cv2.addWeighted(frame, 0.8, trajectory_frame, 1, 0)

            cv2.imshow("Trajectories", combined_frame)     
            #cv2.imshow("Frame", frame_with_detections)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
             
    capture.release()
cv2.destroyAllWindows()
