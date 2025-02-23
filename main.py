import sys

import numpy as np
import pandas as pd
import cv2 as cv
from os import listdir
import re

from LS.LaneSegmentation import LaneSegmentation
from OD.ObjectDetection import ObjectDetection

import warnings
warnings.filterwarnings("ignore")

def main():
    # args = sys.argv[1:]
    # filepath = args[0]
    # filename = args[1]
    filepath = './Source files/'
    fileout_path = './Output/'

    files = listdir(filepath)
    for filename in files:

        # Initializing Lane segmentation algorithm
        ls = LaneSegmentation()

        # Initializing Object detection algorithm
        od = ObjectDetection()

        # Run video algorithm for if the file is not in the below extensions
        if(re.search(r'\bjpg\b|\bpng\b|\bjpeg\b|\bsvg\b|\braw\b|\bdng\b|\bheif\b|\bgif\b', filename) == None):
            # Preparing the video file for reading
            video_file = cv.VideoCapture(filepath+filename)

            # Preparing the video file for writing
            writer = cv.VideoWriter(fileout_path+'out_'+filename, cv.VideoWriter_fourcc(*'mp4v'), round(video_file.get(cv.CAP_PROP_FPS)), (int(video_file.get(3)), int(video_file.get(4))))

            # Total frames in the video for progress bar
            total_frames = int(video_file.get(cv.CAP_PROP_FRAME_COUNT))
            print('Total frames in video: ', total_frames)
            current_frame = 1

            # Reading the frames in the video file
            while(video_file.isOpened()):

                # isFrame, frame
                # isFrame is True if it is a frame. Becomes False at the end of the file
                ret, frame = video_file.read()
                if ret:

                    # Progress bar algorithm
                    percent = round(current_frame/total_frames*100)
                    if percent%5 == 0 :
                        print('\r','#'*round(percent/5),'-'*round((100-percent)/5),f'\t[{percent}%]', sep='', end='')

                    # Lane segmentation pipeline
                    lane_image = ls.pipeline(frame)

                    # Object detection pipeline
                    final_image = od.detectObjects(lane_image)

                    # Writing final output to a file
                    writer.write(final_image)
                    current_frame += 1

                    # cv.imshow('frame', cv.resize(final_image, (640,480)))
                    # if cv.waitKey(1) & 0xFF == ord('q'):
                    #     break
                else:
                    break
            video_file.release()
            writer.release()
            print(f"Completed writing to out_{filename} at {fileout_path}")
            # cv.destroyAllWindows()
        else:
            # Reading image file
            image_file = cv.imread(filepath+filename)

            # Applying lane segmentation
            lane_image = ls.pipeline(image_file)

            # Applying object detection
            final_image = od.detectObjects(lane_image)
            cv.imwrite(f'{fileout_path}out_{filename}', final_image)
            print(f"Completed writing to out_{filename} at {fileout_path}")

if __name__ == '__main__':
    main()