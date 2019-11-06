'''#
import pandas as pd

data = pd.read_csv('C://Users/Mustafa/Desktop/Age-Prediction/sonDataset.csv', index_col=0)

# ilk 5 satırı görüntüle
data.head()
data.tail()
data.shape
'''
import cv2

import os,glob

from os import listdir,makedirs

from os.path import isfile,join
path = 'C://Users/Mustafa/Desktop/Age-Prediction/sample_cropped' # Kaynak dosyası
dstpath = 'C://Users/Mustafa/Desktop/Age-Prediction/sample_greyed' # Fotoların çıkacağı klasör
try:
    makedirs(dstpath)
except:
    print ("Dizin zaten var, görüntüler aynı klasörde yazılacak")

files = [f for f in listdir(path) if isfile(join(path,f))] 
for image in files:
    try:
        img = cv2.imread(os.path.join(path,image))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,gray)
    except:
        print ("{} çevrilemedi".format(image))
for fil in glob.glob("*.jpg"):
    try:
        image = cv2.imread(fil) 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #grileştirme
        cv2.imwrite(os.path.join(dstpath,fil),gray_image)
    except:
        print('{} çevrilemedi')

#####################################################################################################################
images = []
photoNames = []

# egitim verisinin cikti degerleri
def verileriOku(foldername):
    
    for filename in glob.glob(foldername+'/*.jpg'):
        photoNames.append(filename)
        img = cv2.imread(filename)
            
            # resimleri hizli isleyebilmek ve ayni boyuta gelmelerini saglamak icin 200*92 seklinde boyutlandiriyoruz
        img = cv2.resize(img, (64, 64))
            # resimleri siyah beyaz yapiyoruz
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)

image_path = 'images'
verileriOku(image_path)

import numpy as np
images = np.reshape(images, (2476,4096))

import pandas as pd
images_csv = pd.DataFrame(images)
images_csv.to_csv('images.csv')


