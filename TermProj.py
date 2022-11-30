import cv2
import pdb
import glob
import numpy as np
from matplotlib import pyplot as plt

# after this is setup you can do the sobel filtered image and get histograms of that and average them.

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

weatherArr = [cdImg, fgImg, ntImg, rnImg, snImg, suImg]

for i in range(6):
    for img in allPaths[i]:
        n = cv2.imread(img)
        weatherArr[i].append(n)

    print("done path", i + 1)

def createAvgHist(img_list, wtitle):
    histograms = []
    plt.figure(figsize = (16,6))
    ax1 = plt.subplot(1,1,1)
    for j in img_list:
        for i in range(3):
                hist = cv2.calcHist([j], [i], None, [256], [0,256])
                histograms.append(hist)
                
    #pdb.set_trace()
    newarr = np.array_split(histograms, len(img_list))
    newarr = np.asarray(newarr, dtype = object)
    avg_hist = (sum(newarr)) / len(img_list)
    plt.plot(avg_hist[0], color = "b")
    plt.plot(avg_hist[1], color = "g")
    plt.plot(avg_hist[2], color = "r")
    plt.title(wtitle)

Titles = ["Cloudy Histogram", "Foggy Histogram", "Night Histogram", "Rain Histogram", "Snow Histogram", "Sunny Histogram"]

for i in range(6):
    createAvgHist(weatherArr[i], Titles[i])
    print(i + 1, "weather done")

plt.show();

