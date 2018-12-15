# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:00:10 2018
"bubble_counter.py" is intended to assist with counting bubbles observed 
passing through the field of view of a video of sheath flow.

Bubbles are identified through image subtraction and thresholding on the 
"value" channel of the HSV image. Bubbles are counted when they first appear.
The number of bubbles in each frame, the frames containing bubbles, and the
total number of bubbles observed are saved in a datafile. This datafile can be
read and reviewed using "load_bubble_counter_data.py".

@author: Andy
"""

import ImageProcessingFunctions as IPF
import VideoFunctions as VF
import UserInputFunctions as UIF
import Functions as Fun

import skimage.morphology
import skimage.measure
import skimage.color
import skimage.segmentation
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
vidFileStr = 'glyc_co2_1057fps_89us_10V_0250_16bar_79mm_4x.mp4'
checkMask = False

# Processing
# dimensions of structuring element (pixels, pixels)
selem = skimage.morphology.disk(8)
nRefFrame = 0 # frame number of reference/background frame for im subtraction
thresh = 4 # threshold for identifying bubbles, currently heuristic
minSize = 30 # minimum size of object in pixels
skip = 1 # number of frames to jump (1 means analyze every frame)
startFrame = 320
nPixBlurRef = 20
nPixBlur = 5
sizeErode = 6 # number of pixels to erode from mask to remove boundaries
# display
updatePeriod = 1 # update with printout every given number of frames
showResults = True
windowName = 'Video'
waitTime = 100 # milliseconds to wait between frames
screenWidth = 1920

# saving
saveResults = False


###############################################################################
# Processing

# create list of paths to videos
vidPathStr = os.path.join(vidFolder + vidFileStr)
vidPathList = glob.glob(vidPathStr)
nVids = len(vidPathList)

# loop through videos
for v in range(nVids):
    
    # load video data
    vidPath = vidPathList[v]
    vidFileName = Fun.get_file_name_from_path(vidPath)
    Vid = cv2.VideoCapture(vidPath)
    nFrames = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # select reference frame (probably just the first frame)
    refFrame = VF.extract_frame(Vid,nRefFrame)
    # filter frame using mean filter
    refFrame = IPF.mean_filter(refFrame, selem)
    
    # create mask or load from video folder
    maskFile = vidFolder + vidFileName[:-4] + '.pkl'
    maskData = UIF.get_polygonal_mask_data(
            cv2.cvtColor(refFrame, cv2.COLOR_BGR2RGB), 
            maskFile, check=checkMask)
    mask = maskData['mask']  
    # mask image
    refFrame = IPF.mask_image(refFrame, mask)
    
    # Convert reference frame to HSV
    refHSV = cv2.cvtColor(refFrame, cv2.COLOR_BGR2HSV)
    # Only interested in "value" channel to distinguish bubbles
    refValue = refHSV[:,:,2]
    # filter
    refValue = skimage.filters.median(refValue, selem=selem).astype('uint8')
#    refValue = IPF.float2uint8(skimage.filters.gaussian(refValue, sigma=nPixBlurRef)) 

    # prepare to loop through frames 
    # create window to watch results if desired
    if showResults:   
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    # initialize bubble count
    nBubble = 0
    nBubbleCurr = 0
    nBubblePrev = 0
    # save number of bubbles in each frame
    nBubbleInFrame = np.zeros(nFrames)
    # initialize list of numbers of frames with bubbles
    bubbleFramesList = []
    # initialize list of thresholded images of bubbles
    bubbleBWList = []
    # keep track of time
    startTime = time.time()
    
    # loop through frames and count bubbles
    for f in range(startFrame, nFrames, skip):
        # update progress with printouts to console
        if (f%updatePeriod == 0):
            print('Now showing frame #' + str(f))
            print('Number of bubbles seen so far = ' + str(nBubble))
            print('Time elapsed for current video = ' + str(time.time()-startTime))
            
        
        # load current frame
        # image subtraction
        frame = IPF.mask_image(VF.extract_frame(Vid,f), mask)
        # convert to HSV
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # extract "value" channel, which is similar to the intensity
        value = frameHSV[:,:,2]
        # filter
#        value = IPF.float2uint8(skimage.filters.gaussian(value, sigma=nPixBlur))
        
        #
#        print('Average pixel intensity = ' + str(np.mean(value)))
        
        # filter
        value = skimage.filters.median(value,selem=selem).astype('uint8')
        
        
        # identify bubbles
        # TODO debug, replace following lines of code
#        bubbleIm = IPF.highlight_bubbles(value, refValue, thresh, selem=selem,
#                                        min_size=minSize)
        # subtract images
        deltaIm = cv2.absdiff(refValue, value)
        
        # threshold
        threshBW = IPF.threshold_im(deltaIm, thresh)
        # remove objects along edge of mask by "eroding" the mask
        maskEroded = IPF.erode_mask(mask, size=sizeErode)
        clearedEdgeBW = IPF.mask_image(threshBW, maskEroded)
        # smooth out thresholded image
        closedBW = skimage.morphology.binary_closing(clearedEdgeBW, selem=selem)
        # remove small objects
        bubbleBW = skimage.morphology.remove_small_objects(closedBW.astype(bool), min_size=minSize)
        # convert to uint8
        bubbleBW = 255*bubbleBW.astype('uint8')
        
        # count bubbles
        # label remaining objects (presumably bubbles)
        label, nBubbleInFrame[f] = skimage.measure.label(bubbleBW, return_num=True)
        # update count if a new object is seen in the frame
        # this counting method fails when objects enter and exit frame simultaneously,
        # but this is rare when there are few bubbles
        # max(f-1,0) ensures that when f = 0 there is no error
        nNewBubbles = nBubbleInFrame[f]-nBubbleInFrame[max(f-skip,0)]
        # save frames with new bubbles
        if nNewBubbles > 0:
            nBubble += nNewBubbles
            bubbleFramesList += [f]
            # save cleaned up thresholded frame
            bubbleBWList += [bubbleBW]
        
        # display frames in real time during processing
        if showResults:
            # create RGB image of labeled objects
            labeledIm = skimage.color.label2rgb(label, bubbleBW)
            # IPF.scale_brightness(deltaIm)
            # choose images
            im1 = frame 
            im2 = cv2.cvtColor(clearedEdgeBW, cv2.COLOR_GRAY2RGB)
            im3 = cv2.cvtColor(IPF.scale_brightness(deltaIm), cv2.COLOR_GRAY2RGB)
            im4 = cv2.cvtColor(bubbleBW, cv2.COLOR_GRAY2RGB)
            # resize images
            height = im1.shape[0]
            width = im1.shape[1]
            imSize = (int(height*width/(screenWidth/2)),int(screenWidth/2))
            im1 = cv2.resize(im1, imSize)
            im2 = cv2.resize(im2, imSize)
            im3 = cv2.resize(im3, imSize)
            im4 = cv2.resize(im4, imSize)
            # concatenate images
            topIms = np.concatenate((im1,im2), axis=1)
            bottomIms = np.concatenate((im3,im4), axis=1)
            imGrid = np.concatenate((topIms,bottomIms), axis=0)
            # display images
            cv2.imshow(windowName, imGrid)
            
            # waits for allotted number of milliseconds
            k = cv2.waitKey(waitTime)
            # pauses if any key is clicked
            if k != -1:
                break
        
    # free memory of loaded video
    Vid.release()
    
    # report number of bubbles seen in video
    print('Number of bubbles for video ' + str(vidPath) + ' is ' + str(nBubble))
    
    # save data
    # TODO save data file for all videos processed (combined)
    if saveResults:
        data2Save = {}
        data2Save['frames with bubbles'] = bubbleFramesList
        data2Save['bw bubble images'] = bubbleBWList
        data2Save['bubbles in frame'] = nBubbleInFrame
        data2Save['number of bubbles'] = nBubble
        data2Save['skip'] = skip
        # create data path for saving data
        saveDataPath = vidPath[:-4] + '_data.pkl'
        # save list of frames with bubbles
        with open(saveDataPath, 'wb') as f:
            pkl.dump(data2Save, f)