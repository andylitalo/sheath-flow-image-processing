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
import skimage.measure
import skimage.color
from scipy import ndimage
import cv2

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

###############################################################################
# USER PARAMETERS
# Data
# folder containing the video to be processed
vidFolder = '..\\..\\Videos\\'
# name of the video file
vidFile = 'glyc_co2_1057fps_238us_7V_0200_6-5bar_141mm.mp4'
checkMask = True

# Processing
# dimensions of structuring element (pixels, pixels)
selem = skimage.morphology.disk(8)
thresh = 10 # threshold for identifying bubbles
minSize = 20 # minimum size of object in pixels

# display
showResults = True
windowName = 'Video'
waitMS = 10 # milliseconds to wait between frames


###############################################################################
# Processing

# create list of paths to videos
vidPath = os.path.join(vidFolder + vidFile)
vidPathList = glob.glob(vidPath)
nVids = len(vidPathList)

# initialize array to store counts of bubbles
nBubblesArr = np.zeros(nVids)

# loop through videos
for v in range(nVids):
    
    # load video data
    vidPath = vidPathList[v]
    Vid = cv2.VideoCapture(vidPath)
    nFrames = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    # select reference frame (probably just the first frame)
    nRefFrame = 0
    refFrame = VF.extract_frame(Vid,nRefFrame)
    # filter frame using mean filter
    refFrame = IPF.mean_filter(refFrame, selem)
    # create mask or load from video folder
    maskFile = vidFolder + vidFile[:-4] + '.pkl'
    maskData = UIF.get_polygonal_mask_data(refFrame, maskFile, check=checkMask)
    mask = maskData['mask']
    # mask image
    refFrame = IPF.mask_image(refFrame, mask)
    
    # project image onto average color of inner stream
    # average color
    cAve = np.array([np.mean(refFrame[:,:,i]) for i in range(len(refFrame.shape))])
    # normalize
    cAve /= np.linalg.norm(cAve)
    # project reference frame
    refProj = IPF.project_im(refFrame, cAve)
    # median filter to remove saturated pixels
    refProj = skimage.filters.median(refProj, selem=selem)
    
    # loop through video frames   
    cv2.namedWindow(windowName)
    # initialize bubble count
    nBubbles = 0
    for f in range(1270,1290):#nFrames):
        print('Now showing frame #' + str(f))
        # image subtraction
        frame = IPF.mask_image(VF.extract_frame(Vid,f),mask)
        # project frame
        proj = IPF.project_im(frame, cAve)
        # median filter to remove saturated pixels
        proj = skimage.filters.median(proj, selem=selem)
        # subtract images
        deltaIm = cv2.absdiff(refProj, proj)
        
        # threshold
        threshIm = IPF.threshold_im(deltaIm, thresh)
        # smooth out thresholded image
        closedIm = skimage.morphology.binary_closing(threshIm, selem=selem)
        # remove small objects
        cleanIm = skimage.morphology.remove_small_objects(closedIm.astype(bool), min_size=minSize)
        # convert to uint8
        cleanIm = 255*cleanIm.astype('uint8')
        # label remaining objects
        label, nObj = skimage.measure.label(cleanIm, return_num=True)
        # update count
        nBubbles += nObj
        
        # display frames in real time during processing
        if showResults:
            # create RGB image of labeled objects
            labeledIm = skimage.color.label2rgb(label, cleanIm)
            
            # display image
            cleanIm = cleanIm.astype('uint8')
            twoIms = np.concatenate((frame, cv2.cvtColor(IPF.scale_brightness(deltaIm),cv2.COLOR_GRAY2RGB)), axis=1)
            cv2.imshow(windowName, twoIms)
            # waits for allotted number of milliseconds
            k = cv2.waitKey(waitMS)
            # pauses if any key is clicked
            if k != -1:
                break
        
    # free memory of loaded video
    Vid.release()
    # store number of bubbles counted
    nBubblesArr[v] = nBubbles
    print('Number of bubbles for video ' + str(vidPath) + ' is ' + str(nBubbles))