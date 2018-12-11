# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:00:10 2018
"bubble_counter.py" is intended to assist with counting bubbles observed 
passing through the field of view of a video of sheath flow.
@author: Andy
"""

import ImageProcessingFunctions as IPF
import VideoFunctions as VF
import UserInputFunctions as UIF

import skimage.morphology
import cv2

from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

###############################################################################
# USER PARAMETERS
# Video
# folder containing the video to be processed
vidFolder = '..\\..\\Videos\\'
# name of the video file
vidFile = 'glyc_co2_1057fps_238us_7V_0200_6-5bar_141mm.mp4'

# Processing
# dimensions of structuring element (pixels, pixels)
selem = skimage.morphology.disk(10)
thresh = -1

# display
windowName = 'Video'
waitMS = 1000 # milliseconds to wait between frames


###############################################################################
# Processing

# create list of paths to videos
vidPath = os.path.join(vidFolder + vidFile)
vidPathList = glob.glob(vidPath)
nVids = len(vidPathList)

# loop through videos
for v in range(nVids):
    
    # load video data
    vidPath = vidPathList[v]
    Vid = VF.get_video_object(vidPath)
    nFrames = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # select reference frame (probably just the first frame)
    nRefFrame = 0
    refFrame = VF.extract_frame(Vid,nRefFrame)
    # filter frame using mean filter
    refFrame = IPF.mean_filter(refFrame, selem)
    # create mask from reference frame
    mask, boundary = IPF.make_polygon_mask(refFrame)
    refFrame = IPF.mask_image(refFrame, mask)
    
    # loop through video frames
    nFrames = 10
    offset = 1270     
    cv2.namedWindow(windowName)
    for f in range(offset, offset + nFrames):
        print('Now showing frame #' + str(f))
        # image subtraction
        frame = IPF.mask_image(VF.extract_frame(Vid,f),mask)
        # darker image must be second or else change will be 0
        subtIm = cv2.subtract(refFrame, frame)
        # process image
        subtIm = IPF.scale_brightness(subtIm)
        # threshold
        threshIm = IPF.threshold_im(subtIm, thresh, c=1)
        threshIm = cv2.cvtColor(threshIm,cv2.COLOR_GRAY2RGB)
        # display image
        twoIms = np.concatenate((threshIm, subtIm), axis=1)
        cv2.imshow(windowName, twoIms)
        # waits for allotted number of milliseconds
        k = cv2.waitKey(waitMS)
        # pauses if any key is clicked
        if k != -1:
            break