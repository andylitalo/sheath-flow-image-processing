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
import Functions as Fun

import skimage.morphology
import skimage.measure
import skimage.color
import cv2

import numpy as np
import os
import glob
import pickle as pkl

import time

###############################################################################
# USER PARAMETERS
# Data
# folder containing the video to be processed
vidFolder = '..\\..\\Videos\\'
# name of the video file
vidFileStr = 'glyc_co2_1057fps_238us_7V_0200_6-5bar_*mm.mp4'
checkMask = False

# Processing
# dimensions of structuring element (pixels, pixels)
selem = skimage.morphology.disk(8)
thresh = 7 # threshold for identifying bubbles
minSize = 20 # minimum size of object in pixels
skip = 10

# display
updatePeriod = 1000 # update with printout every given number of frames
showResults = False
windowName = 'Video'
waitMS = 10 # milliseconds to wait between frames


###############################################################################
# Processing

# create list of paths to videos
vidPath = os.path.join(vidFolder + vidFileStr)
vidPathList = glob.glob(vidPath)
nVids = len(vidPathList)

# initialize list of frames with bubbles
bubbleFramesList = []

# loop through videos
for v in range(nVids):
    
    # load video data
    vidPath = vidPathList[v]
    vidFileName = Fun.get_file_name_from_path(vidPath)
    Vid = cv2.VideoCapture(vidPath)
    nFrames = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    # select reference frame (probably just the first frame)
    nRefFrame = 0
    refFrame = VF.extract_frame(Vid,nRefFrame)
    # filter frame using mean filter
    refFrame = IPF.mean_filter(refFrame, selem)
    # create mask or load from video folder
    maskFile = vidFolder + vidFileName[:-4] + '.pkl'
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
    
    
    # HSV
    refHSV = cv2.cvtColor(refFrame, cv2.COLOR_BGR2HSV)
    # value
    refValue = refHSV[:,:,2]
    # filter
    refValue = skimage.filters.median(refValue, selem=selem).astype('uint8')

    # prepare to loop through frames    
    if showResults:   
        cv2.namedWindow(windowName)
    # initialize bubble count
    nBubbles = 0
    nBubblesCurr = 0
    nBubblesPrev = 0
    # save number of bubbles in each frame
    nBubblesInFrame = np.zeros(nFrames)
    # keep track of time
    startTime = time.time()
    # loop through frames and count bubbles
    for f in range(0, nFrames, skip):
        if (f%updatePeriod == 0):
            print('Now showing frame #' + str(f))
            print('Number of bubbles seen so far = ' + str(nBubbles))
            print('Time elapsed for current video = ' + str(time.time()-startTime))
        # image subtraction
        frame = IPF.mask_image(VF.extract_frame(Vid,f),mask)
        # convert to HSV
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # extract "value" channel, which is similar to the intensity
        value = frameHSV[:,:,2]
        # filter
        value = skimage.filters.median(value,selem=selem).astype('uint8')
        # subtract images
        deltaIm = cv2.absdiff(refValue, value)
        
        # threshold
        threshIm = IPF.threshold_im(deltaIm, thresh)
        # smooth out thresholded image
        closedIm = skimage.morphology.binary_closing(threshIm, selem=selem)
        # remove small objects
        cleanIm = skimage.morphology.remove_small_objects(closedIm.astype(bool), min_size=minSize)
        # convert to uint8
        cleanIm = 255*cleanIm.astype('uint8')
        # label remaining objects (presumably bubbles)
        label, nBubblesCurr = skimage.measure.label(cleanIm, return_num=True)
        # update count if a new object is seen in the frame
        # this counting method fails when objects enter and exit frame simultaneously,
        # but this is rare when there are few bubbles
        nNewBubbles = nBubblesCurr-nBubblesPrev
        nBubbles += max(nNewBubbles,0)
        # frames with new bubbles
        if nNewBubbles > 0:
            bubbleFramesList += [f]
        # save number of bubbles in frame
        nBubblesInFrame[f] = nBubblesCurr
        # update previous count of number of objects in frame
        nBubblesPrev = nBubblesCurr
        # display frames in real time during processing
        if showResults:
            # create RGB image of labeled objects
            labeledIm = skimage.color.label2rgb(label, cleanIm)
            
            # display image
            cleanIm = cleanIm.astype('uint8')
            # 
            twoIms = np.concatenate((cv2.cvtColor(threshIm, cv2.COLOR_GRAY2RGB), cv2.cvtColor(IPF.scale_brightness(deltaIm),cv2.COLOR_GRAY2RGB)), axis=1)
            cv2.imshow(windowName, twoIms)
            # waits for allotted number of milliseconds
            k = cv2.waitKey(waitMS)
            # pauses if any key is clicked
            if k != -1:
                break
        
    # free memory of loaded video
    Vid.release()
    # report number of bubbles seen in video
    print('Number of bubbles for video ' + str(vidPath) + ' is ' + str(nBubbles))
    data2Save = {}
    data2Save['frames with bubbles'] = bubbleFramesList
    data2Save['bubbles in frame'] = nBubblesInFrame
    data2Save['number of bubbles'] = nBubbles
    # create data path for saving data
    saveDataPath = vidPath[:-4] + '_data.pkl'
    # save list of frames with bubbles
    with open(saveDataPath, 'wb') as f:
        pkl.dump(data2Save, f)