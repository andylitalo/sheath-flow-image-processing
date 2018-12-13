# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 09:26:32 2018

"load_bubble_counter_data.py" reads the datafile produced by "bubble_counter.py"
and allows the user to review the information stored inside, including the
number of bubbles, the number of bubbles in each frame, and the frames with 
bubbles.

@author: Andy
"""

import Functions as Fun
import VideoFunctions as VF

import cv2

import os
import glob
import pickle as pkl


# user parameters
# files and folders
dataFolder = '..\\..\\Videos\\'
vidFolder = '..\\..\\Videos\\'
dataFileStr = 'glyc_co2_1057fps_238us_7V_0200_6-5bar_22mm_data.pkl' 

# display
windowName = 'Frames with Bubbles'
waitTime = 200 # minimum milliseconds to pause at each frame

###############################################################################
# FUNCTIONS
def get_video_name(dataFile, tagLength=9, vidExt='.mp4'):
    """
    Returns the name of the video for the given data file.
    "tagLength" is the number of characters in the tag for the data file,
    including the extension (usually the tag is "_data.pkl", which has 9 chars.)
    """
    return Fun.get_file_name_from_path(dataFile[:-tagLength] + vidExt)
###############################################################################
# SCRIPT

# create file path for data
dataPathStr = os.path.join(dataFolder + dataFileStr)
dataPathList = glob.glob(dataPathStr)
nFiles = len(dataPathList)

# loop through datasets
for d in range(nFiles):
    # load data
    dataPath = dataPathList[d]
    with open(dataPath, 'rb') as f:
        data = pkl.load(f)
    framesBubbles = data['frames with bubbles']
    nBubblesInFrame = data['bubbles in frame']
    
    # create file path to corresponding video
    vidPath = os.path.join(vidFolder + get_video_name(dataPath))
    # load video
    Vid = cv2.VideoCapture(vidPath)
    nFrames = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # set up display
    cv2.namedWindow(windowName)
    cv2.moveWindow(windowName, 40,30)
    # display frames with bubbles, allowing user to click through them
    print('Now showing frames with bubbles.')
    for f in framesBubbles:
        frame = VF.extract_frame(Vid, f)
        print('Showing frame ' + str(f) + ', which contains ' + str(nBubblesInFrame[f]) + ' bubbles')
        cv2.imshow(windowName, frame)
        # waits for allotted number of milliseconds until a key is pressed
        keyIsNotPressed = (cv2.waitKey(waitTime) == -1)
        while keyIsNotPressed:
            keyIsNotPressed = (cv2.waitKey(waitTime) == -1)
            
    # free memory from video when done
    Vid.release()
    # close viewing window
    cv2.destroyAllWindows()