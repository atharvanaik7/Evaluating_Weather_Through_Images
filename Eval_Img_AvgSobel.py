import cv2
import pdb
import glob
import numpy as np
from matplotlib import pyplot as plt

path = glob.glob("Cloudy/*.jpg")
path2 = glob.glob("Foggy/*.jpg")
path3 = glob.glob("Night/*.jpg")
path4 = glob.glob("Rain/*.jpg")
path5 = glob.glob("Snow/*.jpg")
path6 = glob.glob("Sunny/*.jpg")

allPaths = [path, path2, path3, path4, path5, path6]

cdImg = []
fgImg = []
ntImg = []
rnImg = []
snImg = []
suImg = []

cdImgy = []
fgImgy = []
ntImgy = []
rnImgy = []
snImgy = []
suImgy = []

weatherArrgx = [cdImg, fgImg, ntImg, rnImg, snImg, suImg]
weatherArrgy = [cdImgy, fgImgy, ntImgy, rnImgy, snImgy, suImgy]

dsize = (648, 486)

for i in range(6):
    for img in allPaths[i]:
        n = cv2.imread(img)
        n = cv2.resize(n, dsize)
        gx = cv2.Sobel(n, cv2.CV_8U, 1, 0, 3)
        gy = cv2.Sobel(n, cv2.CV_8U, 0, 1, 3)
        
        weatherArrgx[i].append(gx)
        weatherArrgy[i].append(gy)

    print("done path", i + 1)

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
    
    ax1.plot(avg_hist[0], color = "b")
    ax2.plot(avg_histY[0], color = "b")
    ax1.title.set_text(titleX)
    ax2.title.set_text(titleY)

TitlesX = ["CloudyX Line Graph", "FoggyX Line Graph", "NightX Line Graph", "RainX Line Graph", "SnowX Line Graph", "SunnyX Line Graph"]
TitlesY = ["CloudyY Line Graph", "FoggyY Line Graph", "NightY Line Graph", "RainY Line Graph", "SnowY Line Graph", "SunnyY Line Graph"]

for i in range(6):
    createAvgHist(weatherArrgx[i], weatherArrgy[i], TitlesX[i], TitlesY[i])
    print(i + 1, "weather done")

plt.show();

