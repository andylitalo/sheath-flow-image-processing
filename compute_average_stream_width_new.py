# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:39:16 2018

N.B. must type the command "%matplotlib qt" into the console before running,
otherwise will recieve "NotImplementedError" or "Matplotlib is currently using
a non-GUI backend"

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
import pickle as pkl

# User parameters
folder = '..\\..\\DATA\\glyc_in_glyc\\scan_thresh\\'
fileString = 'sheath_glyc_glyc_0372_*.png' #'sheath_cap_0760*.png' # filestring of videos to analyze # glyc: 'sheath_cap_glyc_0100_*.png'
hdr = 'sheath_glyc_glyc_' #'sheath_cap_' # header of filename, including "_" # glyc: 'sheath_cap_glyc_'
border = 2 # number of pixels from edge of outline to start measuring width
widthCapMicron = 800 # inner diameter of capillary in um
outerConversion = 1 # conversion to get actual inner flowrate
pixPerMicron = 1.42 # pixels per micron in image; 1.42 for glyc in glyc; 1.4 for water, 1.475 for glyc; set to 0 to calculate from image by clicking
innerMu = 1.412 # viscosity of inner fluid in Pa.s
outerMu = 1.412 # viscosity of outer fluid in Pa.s
uncertainty = 15 # pixels of uncertainty in stream width
channel = 'g' # channel containing outline of stream (saturated)
eps = 0.1 # inner stream/outer stream radii, small parameter determining meaning of << 
innerFlowRateMin = 0 # [uL/min] smallest flow rate to consider in fitting
# saving parameters
saveData = True
saveFolder = '..\\..\\DATA\\glyc_in_glyc\\data\\'
saveName = 'stream_width_vs_inner_flowrate_glyc.pkl' #'stream_width_vs_inner_flowrate.pkl' # glyc: 'stream_width_vs_inner_flowrate_glyc.pkl'
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
    print('mean stream width is ' + str(streamWidthMean) + \
    ' and stdev of stream width is ' + str(streamWidthStDev))
    # store stream width
    streamWidthList[i] = streamWidthMean
    sigmaList[i] = max(streamWidthStDev, uncertainty) # max between statistical noise and uncertainty

# Combine datapoints for same inner flow rate into one with weighted uncertainty
innerFlowRateList, streamWidthList, sigmaList = Fun.combine_repetitions(innerFlowRateList, streamWidthList, sigmaList)
# shorten outer flow rate list
outerFlowRateList = outerFlowRateList[0]*np.ones([len(innerFlowRateList)])
# ensure that uncertainty doesn't go below the experimental uncertainty
sigmaList[sigmaList < uncertainty] = uncertainty
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
toFit = np.logical_and(innerFlowRateList < eps*outerFlowRateList, \
                        innerFlowRateList >= innerFlowRateMin)
x = innerFlowRateList[toFit]
y = streamWidthMicron[toFit]
sigma = sigmaList[toFit]
# fit power law
m, A, sigmaM, sigmaA = Fun.power_law_fit(x, y, sigma)
xFit = np.linspace(np.min(x), np.max(x), 20)
yFit = A*x**m
plt.plot(x, yFit, 'r--')
# compare to theory
yPredMicron = Fun.stream_width(innerFlowRateList, \
                               outerFlowRateList, widthCapMicron, 
                               innerMu=innerMu, outerMu=outerMu)
plt.plot(innerFlowRateList, yPredMicron, 'b-')
plt.legend(['data','y = (' + str(round(A)) + '+/-' + str(round(sigmaA)) + \
')*x^(' + str(round(m,2)) + '+/-' + str(round(sigmaM,2)) + ')',
            'theory'], loc='best')     
    
if saveData:
    data = {}
    data['inner flow rate'] = innerFlowRateList
    data['outer flow rate'] = outerFlowRateList
    data['stream width'] = streamWidthMicron
    data['uncertainty of inner flow rate'] = sigmaMicronList
    with open(saveFolder + saveName, 'wb') as f:
        pkl.dump(data, f)