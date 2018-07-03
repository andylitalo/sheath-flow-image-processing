# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:39:16 2018

N.B. must type the command "%matplotlib qt" into the console before running,
otherwise will recieve "NotImplementedError"

@author: Andy
"""

# import packages
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import Functions as Fun
import cv2
import skimage.morphology
import skimage.filters
import skimage.feature
from scipy import ndimage
import ImageProcessingFunctions as IPF
import UserInputFunctions as UIF

# User parameters
folder = '..\\DATA\\2018-06-12\\segmentation\\'
fileString = 'sheath_cap_0760_*.png' #'sheath_cap_0760*.png' # filestring of videos to analyze # glyc: 'sheath_cap_glyc_0100_*.png'
hdr = 'sheath_cap_' #'sheath_cap_' # header of filename, including "_" # glyc: 'sheath_cap_glyc_'
border = 2 # number of pixels from edge of outline to start measuring width
widthCapMicron = 556 # inner diameter of capillary in um
outerConversion = 0.86/0.76 # conversion to get actual inner flowrate
pixPerMicron = 1.4 # pixels per micron in image; 1.4 for water, 1.475 for glyc; set to 0 to calculate from image by clicking
uncertainty = 15 # pixels of uncertainty in stream width
channel = 'g' # channel containing outline of stream (saturated)
eps = 0.1 # small parameter determining meaning of <<
# saving parameters
saveData = False
saveFolder = '..\\DATA\\2018-06-12\\segmentation\\'
dataFile = 'stream_width_vs_inner_flowrate_glyc.pkl' #'stream_width_vs_inner_flowrate.pkl' # glyc: 'stream_width_vs_inner_flowrate_glyc.pkl'
# viewing parameters
viewIms = False
maskMsg = 'Click two opposing vertices to define rectangle around portion' + \
' of inner stream desired for analysis.' # message for clicking on mask
# plot parameters
A_FS = 16 # fontsize of axis titles
T_FS = 20 # fontsize of title
MS = 4 # markersize


###############################################################################
# Cleanup existing windows
plt.close('all')
   
# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
# Number of images to consider
nIms = len(fileList)

# initialize 1D data structures to store results
innerFlowRateList = np.zeros([nIms])
outerFlowRateList = np.zeros([nIms])
streamWidthList = np.zeros([nIms]) # list of width of inner stream [pixels]
sigmaList = np.zeros([nIms]) #list of uncertainties in stream width

# Loop through all videos
for i in range(nIms):
    # Parse the filename to get image info
    imPath = fileList[i]
    # Get inner flow rate and store
    innerFlowRateList[i], outerFlowRateList[i] = Fun.get_flow_rates(imPath, hdr, outerConversion=outerConversion)
   
    # Load image
    im = plt.imread(imPath)
    imCopy = np.copy(im) # copy
    # convert copy to 0-255 uint8 image
    imCopy = (255*imCopy).astype('uint8')
    # calculate pixels per micron by clicking on first image if no conversion given
    if i == 0 and pixPerMicron == 0:
        pixPerMicron = UIF.pixels_per_micron(imCopy, widthCapMicron)

    # find edges by determining columns with most saturated (255) pixels
    left, right = IPF.get_edgeX(imCopy, channel=channel)

    # show edges to check that they were found properly
    if viewIms:
        IPF.show_im(imCopy[:,:left,:], 'left edge')
        IPF.show_im(imCopy[:,right:,:], 'right edge')
    
    #### MASKING ###
    imMasked = IPF.create_and_apply_mask(imCopy, 'rectangle', message=maskMsg)
    
    # show masked edges and mask to ensure they were determined properly
    if viewIms:
        IPF.show_im(imMasked[:,:left,:], 'left edge')
        IPF.show_im(imMasked[:,right:,:], 'right edge')
    if viewIms:
        IPF.show_im(mask, 'mask')
        
    # compute stream width and standard deviation 
    streamWidthMean, streamWidthStDev = IPF.calculate_stream_width(imMasked, left, right)
    print 'mean stream width is ' + str(streamWidthMean) + \
    ' and stdev of stream width is ' + str(streamWidthStDev)
    # store stream width
    streamWidthList[i] = streamWidthMean
    sigmaList[i] = max(streamWidthStDev, uncertainty) # max between statistical noise and uncertainty

# Plot stream width as a function of inner flowrate on linear axes
streamWidthMicron = streamWidthList/pixPerMicron
sigmaMicronList = sigmaList / pixPerMicron
plt.figure()
plt.loglog(innerFlowRateList, streamWidthMicron, 'b^', markersize=MS)
plt.errorbar(innerFlowRateList, streamWidthMicron, yerr=sigmaMicronList, fmt='none') # error bars
plt.grid()
plt.xlabel('Inner Flowrate [ul/min]', fontsize=A_FS)
plt.ylabel('Stream width [um]', fontsize=A_FS)
plt.title('Stream width vs. inner flow rate', fontsize=T_FS)
# power-law fit
# only consider small flow rates for which there is a predicted power-law deprightence
small = innerFlowRateList < eps*outerFlowRateList #streamWidthMicron > 0
x = innerFlowRateList[small]
y = streamWidthMicron[small]
sigma = sigmaList[small]
# fit power law
m, A, sigmaM, sigmaA = Fun.power_law_fit(x, y, sigma)
xFit = np.linspace(np.min(x), np.max(x), 20)
yFit = A*x**m
plt.plot(x, yFit, 'r--')
# compare to theory
yPredMicron = Fun.stream_width(innerFlowRateList[small], outerFlowRateList[small], widthCapMicron)
plt.plot(x, yPredMicron, 'b-')
plt.legend(['data','y = (' + str(round(A)) + '+/-' + str(round(sigmaA)) + \
')*x^(' + str(round(m,2)) + '+/-' + str(round(sigmaM,2)) + ')',
            'theory'], loc='best')
# ^what is the problem with the fit?     
       
### THIS THEORY IS NOT RIGHT ###
## linear fit based on exact theory
#xTheory = (streamWidthList/pixPerMicron)**(-2) #convert width to um
#yTheory = 1/innerFlowRateList
#mTheory, bTheory = np.polyfit(xTheory, yTheory, 1)
#outerFlowRateTheory = -1/bTheory
#widthCapMicronTheory = np.sqrt(-mTheory/bTheory)
#print 'theoretical outer flow rate is ' + str(outerFlowRateTheory)
#print 'theoretical width of capillary is ' + str(widthCapMicronTheory)
#plt.figure()
#plt.plot(xTheory, yTheory, 'b^', markersize=MS)
#yTheoryFit = mTheory*xTheory + bTheory
#plt.plot(xTheory, yTheoryFit, 'r--')
#plt.xlabel('1/Inner Flowrate^2 [ul/min]', fontsize=A_FS)
#plt.ylabel('1/Stream width [um]', fontsize=A_FS)
#plt.title('Fit theory', fontsize=T_FS)
#plt.legright(['data', 'Qin^-1 = ' + str(round(mTheory,2)) + 'Din^-2 + ' + str(round(bTheory,3))],
#            loc='best')