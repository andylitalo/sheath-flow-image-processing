# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:00:10 2018
"bubble_counter.py" is intended to assist with counting bubbles observed 
passing through the field of view of a video of sheath flow.
@author: Andy
"""

import ImageProcessingFunctions as IPF
import VideoFunctions as VF
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

###############################################################################
# User parameters
vidFolder = '..\\..\\DATA\\20181210\\'
vidFile = 'glyc_n2_1057fps_238us_7V_0200_6-5bar_141mm.mp4'

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
    Props = VF.parse_video_obj(Vid)
    nFrames = int(Props.NumFrames)
    # select reference frame (probably just the first frame)
    refFrame = VF.extract_frame(Vid,0)
    # show reference frame to check
    plt.figure()
    refFrame = IPF.bgr2rgb(refFrame)
    plt.imshow(refFrame)
    # loop through video frames
#    for f in range(nFrames):
        # image subtraction
        # display image
        # wait for user click to approve going to next image or saving image
