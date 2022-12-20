import cv2
import pdb
import glob
import numpy as np
from matplotlib import pyplot as plt

path = glob.glob("Cloudy/*.jpg")
path2 = glob.glob("Sunny/*.jpg")
path3 = glob.glob("Night/*.jpg")
path4 = glob.glob("Rain/*.jpg")
path5 = glob.glob("Snow/*.jpg")
path6 = glob.glob("Foggy/*.jpg")

allPaths = [path, path2, path3, path4, path5, path6]

clearViewGx = []
clearViewGy = []
poorViewGx = []
poorViewGy = []

clearViewGxF = []
clearViewGyF = []
poorViewGxF = []
poorViewGyF = []

dsize = (1944, 1458)

for i in range(3): # loops through clear view images
    for img in allPaths[i]:
        n = cv2.imread(img) # reads img
        n = cv2.resize(n, dsize) # resize
        n = n[485:973, 647:1297] # crop 1/3
        gx = cv2.Sobel(n, cv2.CV_8U, 1, 0, 3) # apply 3x3 sobel
        gy = cv2.Sobel(n, cv2.CV_8U, 0, 1, 3)
        
        clearViewGx.append(gx)
        clearViewGy.append(gy)

    print("done 3x3 Clear View path", i + 1)

for i in range(3, 6): # loops through poor view images
    for img in allPaths[i]:
        nv = cv2.imread(img)
        nv = cv2.resize(nv, dsize)
        nv = nv[485:973, 647:1297]
        gxp = cv2.Sobel(nv, cv2.CV_8U, 1, 0, 3) 
        gyp = cv2.Sobel(nv, cv2.CV_8U, 0, 1, 3)
        
        poorViewGx.append(gxp)
        poorViewGy.append(gyp)

    print("done 3x3 Poor View path", i + 1)

# ------------------------------------------------------

for i in range(3): # loops through clear view
    for img in allPaths[i]:
        nf = cv2.imread(img) # read img
        nf = cv2.resize(nf, dsize) # downsize
        nf = nf[485:973, 647:1297] # crop 1/3
        gxf = cv2.Sobel(nf, cv2.CV_8U, 1, 0, 5) # apply 5x5 sobel
        gyf = cv2.Sobel(nf, cv2.CV_8U, 0, 1, 5)
        
        clearViewGxF.append(gxf)
        clearViewGyF.append(gyf)

    print("done 5x5 Clear View path", i + 1)

for i in range(3, 6): # loops through poor view
    for img in allPaths[i]:
        npf = cv2.imread(img)
        npf = cv2.resize(npf, dsize)
        npf = npf[485:973, 647:1297]
        gxpf = cv2.Sobel(npf, cv2.CV_8U, 1, 0, 5)
        gypf = cv2.Sobel(npf, cv2.CV_8U, 0, 1, 5)
        
        poorViewGxF.append(gxpf)
        poorViewGyF.append(gypf)

    print("done 5x5 Poor View path", i + 1)

# ------------------------------------------------------

def createAvgHist(img_listX, img_listY, titleX, titleY):
    histogramsX = []
    histogramsY = []
    plt.figure(figsize = (16,6))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    for j in img_listX:
            hist = cv2.calcHist([j], [0], None, [256], [0,256])
            histogramsX.append(hist)
    for i in img_listY:
            histY = cv2.calcHist([i], [0], None, [256], [0,256])
            histogramsY.append(histY)
                
    #pdb.set_trace()
    newarr = np.array_split(histogramsX, len(img_listX))
    newarr = np.asarray(newarr, dtype = object)
    avg_hist = (sum(newarr)) / len(img_listX)

    newarrY = np.array_split(histogramsY, len(img_listY))
    newarrY = np.asarray(newarrY, dtype = object)
    avg_histY = (sum(newarrY)) / len(img_listY)
    
    ax1.hist(avg_hist[0])
    ax2.hist(avg_histY[0])
    ax1.title.set_text(titleX)
    ax2.title.set_text(titleY)

createAvgHist(clearViewGx, clearViewGy, "Clear View Gx 3x3 Line Graph", "Clear View Gy 3x3 Line Graph")
print("Clear View 3x3 Done")
createAvgHist(poorViewGx, poorViewGy, "Poor View Gx 3x3 Line Graph", "Poor View Gy 3x3 Line Graph")
print("Poor View 3x3 Done")
createAvgHist(clearViewGxF, clearViewGyF, "Clear View Gx 5x5 Line Graph", "Clear View Gy 5x5 Line Graph")
print("Clear View 5x5 Done")
createAvgHist(poorViewGxF, poorViewGyF, "Poor View Gx 5x5 Line Graph", "Poor View Gy 5x5 Line Graph")
print("Poor View 5x5 Done")

plt.show();

