# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:44:10 2018

@author: Andy
"""

# import packages
import glob
import os
import VideoFunctions as VF
import matplotlib.pyplot as plt
import numpy as np
import Functions as Fun
import cv2
import skimage.morphology
import skimage.filters
import skimage.feature
from scipy import ndimage

# User Parameters
# data for video
folder = '..\\..\\Videos\\' # folder containing videos
fileString = 'glyc_n2_1057fps_238us_7V_0200_6-5bar_141mm.mp4'#'8819fps_108us_16mLmin.mp4' # filestring of videos to analyze
fps = 0 # set to 0 if unknown and program will obtain fps from video object
# analysis parameters
startFrame = 1
endFrame = 100
subtractRef = True
meanFilter = True
kernel = np.ones((5,5),np.float32)/25 # kernel for gaussian filter
thresh = 5
# Structuring element is radius 2 disk
selem = skimage.morphology.disk(10)
nDilations = 0
showIms = False
minSize = 250
# saving parameters
saveIms = True
saveFolder = '..\\Videos\\8819fps_108us_16mLmin_video\\'



###############################################################################
# Cleanup existing windows
plt.close('all')
   
# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
# Number of videos to consider
nVideos = len(fileList)

# Loop through all videos
for v in range(nVideos):
    # Parse the filename to get video info
    videoPath = fileList[v]
    # Create video "object" for manipulation
    vid = VF.get_video_object(videoPath)
    nFrames = int(vid.get(7)) # number of frames in video
    # Loop through designated frames that are available
    lastFrame = np.min([nFrames, endFrame])
    if startFrame >= lastFrame:
        print('start frame is larger than last frame.')
        continue
    # save first frame as reference frame for image subtraction
    vid.set(1,0)
    ret, refFrame = vid.read()
    # smooth reference frame to reduce noise
    if meanFilter:
        for i in range(3):
            refFrame[:,:,i] = skimage.filters.rank.mean(refFrame[:,:,i], selem)
    # show reference frame to make sure there are no features
    plt.figure()
    plt.imshow(refFrame)
    # loop through all frames
    for f in range(startFrame, lastFrame):
        # Extract frame from video--frame is a matrix of pixel values
        vid.set(1,f)
        ret, frame = vid.read()
        # show frame
        if showIms:
            plt.figure()
            plt.imshow(frame)
        # apply mean filter to frame
        if meanFilter:
            for i in range(3):
                frame[:,:,i] = skimage.filters.rank.mean(frame[:,:,i], selem)
        # extract green channel
        frameG = frame[:,:,1]
        refFrameG = refFrame[:,:,1]
        # subtract reference frame from frame
        if subtractRef:
            frame1 = frameG.astype(int)
            refFrame1 = refFrameG.astype(int)
            frame1 = refFrame1 - frame1
            frame1[frame1<0] = 0
            frameG = frame1.astype('uint8')
        # threshold
        ret, bubble = cv2.threshold(frameG,thresh,255,cv2.THRESH_BINARY)
        # close bubble
        bubbleClosed = skimage.morphology.binary_closing(bubble, selem=selem)
        # fill holes
        bubbleFilled = ndimage.morphology.binary_fill_holes(bubbleClosed)
        # remove fringes
        bubbleFiled = skimage.morphology.binary_opening(bubbleFilled, selem=selem)
        # remove small objects
        bubbleFilled = skimage.morphology.remove_small_objects(bubbleFilled, min_size=minSize)
        # find edge of bubble using Canny filter
        bubbleEdge = skimage.feature.canny(bubbleFilled.astype(float), 1.4)
        # overlay edge with image in red channel
        frameR = frame[:,:,1]
        frameR[bubbleEdge] = 255
        frame[:,:,1] = frameR
        # convert boolean image to 255 scale
        bubbleFilled= bubbleFilled.astype('uint8')
        bubbleFilled *= 255
        # show final result
        if showIms:
            plt.figure()
            plt.imshow(bubbleFilled)
            plt.pause(0.1)
        # save image with edge overlayed
        if saveIms:
            cv2.imwrite(saveFolder + fileString[:-4] + '_' + str(f) + '.png', frame)