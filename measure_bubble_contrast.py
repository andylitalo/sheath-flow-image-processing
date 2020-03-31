# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:00:10 2018
"measure_bubble_contrast.py" is intended to assist with counting bubbles observed 
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
import skimage.feature
import cv2
from scipy import ndimage

import numpy as np
import os
import glob
import pickle as pkl
import matplotlib.pyplot as plt

import time

###############################################################################
# USER PARAMETERS
# Data
# folder containing the video to be processed
vidFolder = '..\\..\\..\\EXPERIMENTS\\foaming_polyol_co2\\20200307_glyc_co2_v360\\'
# name of the video file
vidFileStr ='glyc_co2_v360_*.mp4'

# Processing
# dimensions of structuring element (pixels, pixels)
selem = skimage.morphology.disk(8)
refFrameList = [0, 0, 11, 0, 9, 4, 7, 12, 4, 4, 0, 0, 0] # frame number of reference/background frame for im subtraction
thresh = 10#5 # threshold for identifying bubbles, currently heuristic
minSize = 13 # minimum size of object in pixels
skip = 1 # number of frames to jump (1 means analyze every frame)
startFrame = 0
sizeErode = 6 # number of pixels to erode from mask to remove boundaries
frameWidth = 20 # width of frame to remove from edges
firstVid = 3 # first video to process
lastVid = 7 # last video to process

# display
showResults = False
updatePeriod = 50 # update with printout every given number of frames
windowName = 'Video'
waitTime = 500 # milliseconds to wait between frames
screenWidth = 1920 # width of computer screen in pixels

# saving
saveResults = True
saveFolder = vidFolder + 'results\\'
saveName = 'gradient_data.pkl'

###############################################################################
# FUNCTIONS

def get_exp_time(fileName):
    """Extracts the exposure time [us] from the video name."""
    # get index of the end of the directory name
    i = fileName.rfind('\\')
    return int(fileName[i+20:i+23])

def get_fps(fileName):
    """
    Returns the frames per second from the current filename (depends on naming
    convention).
    """
    # get index of the end of the directory name
    i = fileName.rfind('\\')
    return int(fileName[i+15:i+19])

def get_power(fileName):
    """Extracts the power of the light source [W] from the video name."""
    # get index of the end of the directory name
    i = fileName.rfind('\\')
    return int(fileName[i+36:i+39])
  
def get_number(fileName):
    """Returns the chronological number of the video."""
        # get index of the end of the directory name
    i = fileName.rfind('\\')
    j = fileName.rfind('.mp4')
    return int(fileName[i+40:j])

###############################################################################
cv2.destroyAllWindows()

# Processing

# creates list of paths to videos
vidPathStr = os.path.join(vidFolder + vidFileStr)
vidPathList = glob.glob(vidPathStr)
nVids = len(vidPathList)

# initializes lists of parameters to save
expTimeList = []
powerList = []
vidNameList = []
meanGradList = []

# loop through videos
for v in range(nVids):
    
    # load video data
    vidPath = vidPathList[v]
    vidFileName = Fun.get_file_name_from_path(vidPath)
    # loads video with OpenCV
    Vid = cv2.VideoCapture(vidPath)
    nFrames = int(Vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # gets reference frame number based on video number in case glob mixes order
    num = get_number(vidFileName)
    if num < firstVid or num > lastVid:
        continue

    # stores video file name
    vidNameList += [vidFileName]
    nRefFrame = refFrameList[num]
    fps = get_fps(vidFileName)
    
    # select reference frame if given
    if nRefFrame >= 0:
        refFrame = VF.extract_frame(Vid,nRefFrame)
        # filter frame using mean filter
        refFrame = IPF.mean_filter(refFrame, selem)
    # otherwise generate reference frame by averaging 100 frames
    else:
        refFrame = IPF.average_frames(Vid)
    
    # converts reference frame to HSV
    refHSV = cv2.cvtColor(refFrame, cv2.COLOR_BGR2HSV)
    # extracts "value" channel since most useful for distinguishing brightness variation
    refValue = refHSV[:,:,2]
    # applies median filter
    refValue = skimage.filters.median(refValue, selem=selem).astype('uint8')

    # prepare to loop through frames 
    # create window to watch results if desired
    if showResults:   
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    # keep track of time
    startTime = time.time()
    # collects average gradients above 0
    posGradList = []
    
    # loop through frames and count bubbles
    for f in range(startFrame, nFrames, skip):
        # update progress with printouts to console
        if (f%(skip*updatePeriod) < skip):
            print('Now showing frame #' + str(f))
            print('Time elapsed for analysis of current video: ' + str(time.time()-startTime) + ' s.')
            
        # extracts frame
        frame = VF.extract_frame(Vid,f)
        # converts to HSV
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # extract "value" channel, which is similar to the intensity
        value = frameHSV[:,:,2]
        # applies median filter
        value = skimage.filters.median(value,selem=selem).astype('uint8')
        # subtracts images
        deltaIm = cv2.absdiff(refValue, value)
        
        # thresholds image
        threshBW = IPF.threshold_im(deltaIm, thresh)
        # smooths out thresholded image
        closedBW = skimage.morphology.binary_closing(threshBW, selem=selem)
        # removes small objects
        bubbleBW = skimage.morphology.remove_small_objects(closedBW.astype(bool), min_size=minSize)
        # converts to uint8
        bubbleBW = 255*bubbleBW.astype('uint8')
        
        # finds outer boundary of thresholded bubble using Canny filter
        bubbleFilled = ndimage.morphology.binary_fill_holes(bubbleBW)
        bubbleEdge = 255*skimage.feature.canny(bubbleFilled.astype(float), 1.4)
        # converts to uint8
        bubbleEdge = bubbleEdge.astype('uint8')
        # remove any white pixels slightly inside the border of the image
        fullMask = np.ones([bubbleEdge.shape[0]-2*frameWidth, 
                            bubbleEdge.shape[1]-2*frameWidth])
        deframeMask = np.zeros_like(bubbleEdge)
        deframeMask[frameWidth:-frameWidth,frameWidth:-frameWidth] = fullMask
        bubbleEdge = IPF.mask_image(bubbleEdge, deframeMask)
            
         # computes magnitude of gradient of bkg-subt image at each pixel
        # using the Scharr filter from OpenCV--apparently more accurate
        # than the Sobel filter for 3 x 3 convolutions
        x_grad = cv2.Scharr(deltaIm, cv2.CV_64F, 1, 0)
        y_grad = cv2.Scharr(deltaIm, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt( x_grad**2 + y_grad**2 )
        grad_scaled = grad_mag #grad_mag.astype(float) / np.max(grad_mag) * 255.
        grad = grad_scaled.astype('uint8')
        # only extracts the gradient values along the edge pixels
        gradMasked = IPF.mask_image(grad, bubbleEdge)
        # stores the average gradient if it is greater than 0
        rows, cols = np.nonzero(gradMasked)
        m = np.mean(gradMasked[rows,cols])
        if m > 0:
            posGradList += [m]
            
        # display frames in real time during processing
        if showResults: 
            # choose images
            im1 = frame 
            im2 = cv2.cvtColor(bubbleBW, cv2.COLOR_GRAY2RGB)
            im3 = cv2.cvtColor(bubbleEdge, cv2.COLOR_GRAY2RGB)
            im4 = cv2.cvtColor(gradMasked, cv2.COLOR_GRAY2RGB)
            
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
     
    # averages the gradients after looping through all frames and stores in list
    meanGradList = np.mean(np.array(posGradList))
    # free memory of loaded video
    Vid.release()
    
# saves data
# TODO save data file for all videos processed (combined)
if saveResults:
    data2Save = {}
    data2Save['ref frames'] = refFrameList
    data2Save['exposure times [us]'] = expTimeList
    data2Save['powers [W]'] = powerList
    data2Save['file names'] = vidNameList
    data2Save['mean gradients [pixels]'] = meanGradList
    # other analys parameters
    data2Save['skip'] = skip
    data2Save['threshold'] = thresh
    data2Save['minimum size (pixels)'] = minSize

    # saves data to pickle file
    with open(saveFolder + saveName, 'wb') as f:
        pkl.dump(data2Save, f)