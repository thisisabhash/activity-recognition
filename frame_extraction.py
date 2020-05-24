import cv2
import math
import datetime
import os
import shutil
import config


def main():
    
    # print start time
    print('Frame extraction start: ' + str(datetime.datetime.now()))
    
    for i in range(1, config.NUM_VIDEOS + 1):
        
        # form video file path
        video_path = '{}/video{}.mp4'.format(config.DATA_DIR, str(i).rjust(2, '0'))
        print(video_path)
        if not os.path.isfile(video_path):
            print('File: ' + str(video_path) + ' does not exist');
            continue
        
        # sanity check to remove previously created folders
        # if this code is run multiple times
        video_directory = '{}/video{}'.format(config.DATA_DIR, str(i).rjust(2, '0'))
        if os.path.isdir(video_directory):
            shutil.rmtree(video_directory)
        os.mkdir(video_directory)
        
        # counter for frames
        frame_count = 0
        
        # get VideoCapture object from the video
        video_cap = cv2.VideoCapture(video_path)   
        
        # frame rate
        frame_rate = video_cap.get(cv2.CAP_PROP_FPS) 
        #print('frame rate: ' + str(frameRate))
        
        while(video_cap.isOpened()):
            # get frame index
            frame_index = video_cap.get(cv2.CAP_PROP_POS_FRAMES) 
            return_val, frame = video_cap.read()
        
            if (return_val != True):
                # No frames to read
                break
            
            # sample frames at 1fps, videos in cholec80 are captured at 25fps
            if (frame_index % math.floor(frame_rate) == 0):
                image_path = ('{}/image{}.jpg').format(video_directory, str(frame_count))
                #print('image_path:' + image_path)
                
                frame_count += 1
                ret = cv2.imwrite(image_path, frame)
                
        video_cap.release()
    
    # print end time
    print ('Frame extraction done: ' + str(datetime.datetime.now()))

if __name__ == '__main__':
    main()