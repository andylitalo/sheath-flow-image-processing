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
# Video
# folder containing the video to be processed
vidFolder = '..\\..\\Videos\\'
# name of the video file
vidFile = 'glyc_co2_1057fps_238us_7V_0200_6-5bar_141mm.mp4'

# Processing
# dimensions of structuring element (pixels, pixels)
selem = skimage.morphology.disk(10)
thresh = 127 # threshold for identifying bubbles
minSize = 20 # minimum size of object in pixels

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
    Vid = cv2.VideoCapture(vidPath)
    nFrames = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # select reference frame (probably just the first frame)
    nRefFrame = 0
    refFrame = VF.extract_frame(Vid,nRefFrame)
    # filter frame using mean filter
    refFrame = IPF.mean_filter(refFrame, selem)
    # create mask from reference frame
    mask, boundary = IPF.make_polygon_mask(refFrame)
    refFrame = IPF.mask_image(refFrame, mask)
    
    # project image onto average color of inner stream
    # average color
    cAve = np.array([np.mean(refFrame[:,:,i]) for i in range(len(refFrame.shape))])
    # normalize
    cAve /= np.linalg.norm(cAve)
    # project reference frame
    refProj = IPF.project_im(refFrame, cAve)
    
    # loop through video frames
    nFrames = 10
    offset = 1270     
    cv2.namedWindow(windowName)
    # initialize bubble count
    nBubbles = 0
    for f in range(offset, offset + nFrames):
        print('Now showing frame #' + str(f))
        # image subtraction
        frame = IPF.mask_image(VF.extract_frame(Vid,f),mask)
        # project frame
        proj = IPF.project_im(frame, cAve)
        # subtract images
        deltaIm = IPF.scale_brightness(cv2.absdiff(refProj, proj))
        
        # threshold
        threshIm = IPF.threshold_im(deltaIm, thresh)
        # smooth out thresholded image
        # close bubble
        # TODO: something's wrong here************8
        closed = skimage.morphology.binary_closing(threshIm, selem=selem)
        # fill holes
        filled = ndimage.morphology.binary_fill_holes(closed)
        # remove fringes
        noFringe = skimage.morphology.binary_opening(filled, selem=selem)
        # remove small objects
        cleanIm = skimage.morphology.remove_small_objects(noFringe, min_size=minSize)
        
        # label remaining objects
        label, nObj = skimage.measure.label(cleanIm, return_num=True)
        # update count
        nBubbles += nObj
        
        # create RGB image of labeled objects
        labeledIm = skimage.color.label2rgb(label, cleanIm)
        
        # display image
        cleanIm = cleanIm.astype('uint8')
        twoIms = np.concatenate((cv2.cvtColor(threshIm,cv2.COLOR_GRAY2RGB), cv2.cvtColor(closed.astype('uint8'),cv2.COLOR_GRAY2RGB)), axis=1)
        cv2.imshow(windowName, twoIms)
        # waits for allotted number of milliseconds
        k = cv2.waitKey(waitMS)
        # pauses if any key is clicked
        if k != -1:
            break
        
#Vid.release()
#cv2.destroyAllWindows()