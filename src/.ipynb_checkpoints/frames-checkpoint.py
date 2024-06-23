import os 
import cv2

video_directory = './Video_Search_Assignment/Downloads/YouTube-Videos/'
save_directory ='./Video_Search_Assignment/Downloads/Frames/'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)


video_files =[f for f in os.listdir(video_directory) if f.endswith('.mp4)')]

for video_file in video_files:
    video_path = os.path.join(video_directory, video_file)
    
    new_save_directory =  os.path.join(save_directory, os.path.splitext(video_file)[0])
    if not os.path.exists(new_save_directory):
        os.makedirs(new_save_directory)

    capture = cv2.VideoCapture(video_path)
    
    frame_number = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        
        save_path = os.path.join(new_save_directory, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(save_path,frame)
        print(f"Saved {save_path}")
        
        frame_number += 1
    
    capture.release()
    
cv2.destroyAllWindows()
    
    
