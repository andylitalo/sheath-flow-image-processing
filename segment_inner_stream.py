# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:44:10 2018

@author: Andy

NOTE: Must type "%matplotlib qt" into console to run file in python(x,y)
(this ensures that matplotlib windows open in a new window rather than inside
the console output). Otherwise you will receive a "NotImplementedError."
"""

# import packages
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.morphology
import skimage.filters as filters
import UserInputFunctions as UIF
import ImageProcessingFunctions as IPF
from scipy.stats import mode

# User Parameters
# data for video
folder = '..\\..\\DATA\\glyc_in_glyc\\' # folder containing videos
fileString = 'sheath_glyc_glyc_0372_*_d1_*.jpg' # filestring of videos to analyze, glycerol: 'sheath_cap_glyc_0100*.jpg'
bfFile = 'brightfield_d1.jpg' #image of bright field, light but no flow
maskMsg = 'Click opposing corners of rectangle to include desired section of image.'
maskDataFile = 'maskData_glyc_glyc_20180703.pkl'#'maskData_180613.pkl' # glycerol: 'maskData_glyc_180620.pkl'
# analysis parameters
meanFilter = True
kernel = np.ones((5,5),np.float32)/25 # kernel for gaussian filter
#bPct = 0#55 # percentile of pixels from blue channel kept glycerol: 45
#rNegPct = 80 # percentile of pixels from negative of red channel kept glycerol: 60
#rNegThresh = 160
#thresh = 214
threshWindow = 15 # number of values above and below threshold to compute
skip = 5
#frac = 0.03 # percent of peak of histogram to cutoff with thresholding
#tol = 1 # number of pixels of change in threshold that are tolerated for convergence of automatic thresholding
autoThresh = True # automatically determines threshold for the image if True
# TODO automate selection of these colors
streamRGB = np.array([144,178,152]) # rgb values for predominant color in inner stream
bkgdRGB = np.array([255,211,163])
# Structuring element is radius 10 disk
selem = skimage.morphology.disk(10)
showIm = False
showCounts = False # show counts of number of pixels with each value
minSize = 10000
# saving parameters
saveIm = True
saveFolder = '..\\..\\DATA\\glyc_in_glyc\\scan_thresh\\'


###############################################################################
# Cleanup existing windows
plt.close('all')

# Load bright field as reference for background subtraction
bf = plt.imread(folder + bfFile)
# apply gaussian filter
bf = IPF.rgb_gauss(bf, selem)

# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
# Number of images to consider
nIms = len(fileList)

# Loop through all videos
for i in range(7,nIms):
    ### EXTRACT AND SMOOTH IMAGE ###
    # Parse the filename to get image info
    imPath = fileList[i]
    # Load image and create copy to prevent alteration of original
    im = plt.imread(imPath)
    if showCounts:
        IPF.show_im(im, 'image', showCounts=showCounts)
    # copy and apply mean filter to each channel (rgb) of image
    imCopy = IPF.rgb_gauss(np.copy(im), selem)

    ### CORRECT BRIGHTFIELD INHOMOGENEITIES ###
    imCopy = IPF.scale_by_brightfield(imCopy, bf)
    if showIm:
        IPF.show_im(imCopy, 'brightfield scaled')
    ### MASK IMAGE ###
    # user-defined mask for determining where to search
    maskData = UIF.get_rect_mask_data(imCopy, maskDataFile)
    mask = maskData['mask']
    roiLims = maskData['xyMinMax']
    roi = IPF.get_roi(imCopy, roiLims)

    ### THRESHOLD PROJECTION OF IMAGE ONTO STREAM COLOR ###
    imProj = IPF.get_channel(roi,'custom',rgb=streamRGB, subtRGB=bkgdRGB)
    # apply mean filter to smooth out oscillations in pixel histogram
    imProj = filters.rank.mean(imProj, selem)
    if showCounts:
        IPF.show_im(imProj, title='projected image', showCounts=showCounts)
#    ### THRESHOLD NEGATIVE OF RED CHANNEL TO IDENTIFY STREAM ###
#    imRNeg = IPF.get_negative(IPF.get_channel(roi,'r'))
    if autoThresh:
#        thresh = IPF.get_auto_thresh_hist(imProj, frac=frac)
        thresh = IPF.get_auto_thresh_double_gaussian(imProj, showPlot=showIm)
#        thresh = IPF.get_auto_thresh_plateau(imProj, frac=frac)
    
    ### LOOP THROUGH DIFFERENT VALUES OF THRESHOLD NEAR COMPUTED VALUE ###
    for thr in range(thresh-threshWindow, thresh+threshWindow+1,skip):
        ret, imInnerStream = cv2.threshold(imProj,thr,255,cv2.THRESH_BINARY)
        if showIm:
            IPF.show_im(imInnerStream, 'inner stream')
    
        ### CLEAN UP BINARY IMAGE ###
        imFilled = IPF.clean_up_bw_im(imInnerStream, selem, minSize)
        if showIm:
            IPF.show_im(imFilled,'Filled holes of Inner Stream image')
    
        ### TRACE CONTOUR OF LARGEST OBJECT ###
        imCntRoi = IPF.get_contour_bw_im(imFilled, showIm)
        imSuperimposed, ret = IPF.superimpose_bw_on_color(im, imCntRoi, roiLims,channel='g')
        # place contour in full size image and skip images with no contour
        if not ret:
            print('no contour found in ' + imPath)
            continue
        # show final result
        if showIm:
            IPF.show_im(imSuperimposed,'Image with outline')
    
        ### SAVE IMAGE ###
        # save image with edge overlayed
        if saveIm:
            # save image with outline overlaid for viewing
            dirTokens = imPath.split('\\')
            fileName = dirTokens[-1]
            saveName = saveFolder + fileName[:-4] + '_thr' + str(thr) + '.png'
            # save in save folder as .png rather than previous file extension
            cv2.imwrite(saveName, cv2.cvtColor(imSuperimposed, cv2.COLOR_RGB2BGR))
            print('Saved ' + str(i+1) + ' of ' + str(nIms) + ' images.')
