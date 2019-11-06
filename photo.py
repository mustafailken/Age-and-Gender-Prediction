# -*- coding: utf-8 -*-
#Fotoğraftan cinsiyet ve yaş tahmini yapılması
#Kütüphanelerin eklenmesi.

import numpy as np
import cv2
import os
import pickle
import math
import keras.models

#Yazdırma dökümanının oluşturulması.
print(__doc__)

def find_marker(image):
	# Fotoğrafın gri tonlamaya dönüştürülmesi, bulanıklaştırılması ve kenarlarının tespit edilmesi.
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)

	# Kenar görüntüsündeki şekillerin bulunması ve en büyüğünün korunması.
	_, cnts, _= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	c = max(cnts, key = cv2.contourArea)

	# Kağıt bölgesinin sınırlayıcı değerini hesaplar ve döndürür
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# Hedefin kameraya olan uzaklığının hesaplanması.
	return (knownWidth * focalLength) / perWidth

KNOWN_DISTANCE = 14

KNOWN_WIDTH = 5


# Kullanacağımız fotoğrafların listesinin başlatılması.
image = cv2.imread("C://Users/Mustafa/Desktop/deneme.jpg")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

clf1 = keras.models.load_model("C://Users/Mustafa/Desktop/Age-Prediction/Gender/yeniModel.h5py")

clf = pickle.load(open("C://Users/Mustafa/Desktop/Age-Prediction/clf4.pkl","rb"))

#sınıflandırıcı dosyaların yüklenmesi.
face_cascade=cv2.CascadeClassifier('C://Users/Mustafa/Desktop/Age-Prediction/Haarcascades_Datasets/haarcascade_frontalface_default.xml ')

eye_cascade=cv2.CascadeClassifier('C://Users/Mustafa/Desktop/Age-Prediction/Haarcascades_Datasets/haarcascade_eye.xml ')



#Kamera döngüsü.
while 1:
        img = cv2.imread('C://Users/Mustafa/Desktop/Age-Prediction/sample_images/3382_1981-12-02_2013.jpg')

        img = cv2.resize(img,(64,64))

        bailey = np.expand_dims(img, axis=0)
        
        prediction_b = clf1.predict(bailey)

        if math.floor(prediction_b) >=0.15:

                prediction_b = "Male"
                
        else:
                prediction_b = "Female"
        
        print(prediction_b)
        
        #Gri tonlama dönüşümünün yapılması.
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #Yüz koordinatlarının tespitinin yapılması.
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
       #Dikdörtgenlerin oluşturulması.
        for (x,y,w,h) in faces:
                
                #Yüz bölgesi için
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                
                #Yüz kısmının çıkarılması.
                roi_gray = gray[y:y+h, x:x+w]
                
                roi_color = img[y:y+h, x:x+w]
                
                #Tahmin için yeniden şekillendirme yapılması.
                simg = cv2.resize(roi_gray,(10,10))
                
                #Pürüzlerin düzleştirilmesi.
                simg = simg.flatten().reshape(-1,1)
                
                #Transpozunun alınması.
                simg = simg.T/10.0
                                
                #Değerin tahmin edilmesi.
                res = clf.predict(simg)
                
#                if res//2 >15:
#                       print("Cinsiyet :{}\tTahmin edilen yaş :{}".format((prediction_b),abs(res)//2))
                print("Cinsiyet :{}\tTahmin edilen yaş :{}".format((prediction_b),res))   
                #Gözlerin tespitinin yapılması.
                eyes = eye_cascade.detectMultiScale(roi_gray)

                marker = find_marker(roi_color)
                
                inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
                
                cv2.putText(img, "%.2fft" % (inches / 12),(x , y), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 255), 1)

                
                #Göz koordinatlarının döngüsünün yapılması.
                for (ex,ey,ew,eh) in eyes:
                        
                        #Dikdörtgenlerin oluşturulması.
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #Görüntünün görüntülenmesi.
        cv2.imshow('img',img)
        
        #Bekleme tuşunun oluşturulması.
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
#Kameranın serbest bırakılması.

#Pencerenin kapatılması.
cv2.destroyAllWindows()
