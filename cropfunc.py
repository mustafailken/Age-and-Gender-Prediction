
#Kütüphanelerin eklenmesi.

import pandas as pd
import numpy as np
import cv2
import os

class Imagescale(object):

    #Classmethod un çağrılması ve görüntü yeniden boyutlandırma işlevini tanımlama.
    @classmethod
    def resizeimage(self):

        #Veri kümesini ayıklamak için dzini değiştirme.
        os.chdir("C://Users/Mustafa/Desktop/Age-Prediction/sample_images")
        
        #Görüntü içeren tüm klasörlerin listesi.
        dir = os.listdir()
        
        #Tüm dizinler ve görüntüler arasında döngü kurulması.
        for i in dir:

            #Hepsinin listelenmesi.
            list = os.listdir()

            #Her dizindeki görüntü için.
            for j in list:
                
                #Görüntünün okunması.
                img = cv2.imread(j)
                
                #Görüntünün yeniden boyutlandırılması.
                rm = cv2.resize(img,(64,64))
                
                #Yeniden boyutlandırılan fotoğrafın kaydedilmesi.
                cv2.imwrite('C://Users/Mustafa/Desktop/Age-Prediction/sample_images'+str(j),rm)
        
        #Ana veri kümesi klasörüne dönülmesi.
        return os.chdir('C://Users/Mustafa/Desktop/Age-Prediction')


    #Classmethod un çağrılması ve kırpma işlevinin tanımlanması.
    @classmethod
    def detectfaces_crop(self):
                        
        #Tüm görüntünün kaldırıldığı ve yeniden boyutlandırıldığı dizin konumu.

        os.chdir("C://Users/Mustafa/Desktop/Age-Prediction/sample_images")
                        
       #Tüm görüntülerin listesinin oluşturulması.
        images = os.listdir()
                        
        #Görüntülerdeki yüzlerin algılanması.
                        
        #Veri kümesinin dizin konumu.
        face_cascade=cv2.CascadeClassifier('C://Users/Mustafa/Desktop/Age-Prediction/Haarcascades_Datasets/haarcascade_frontalface_default.xml')
                        
        for image in images:
                        
            #Görüntüyü okumak.
            img = cv2.imread(image)
                        
            #Gri tonlamalı dönüşümün yapılması.
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
            
            #Yüz koordinatlarının belirlenmesi. 
            faces=face_cascade.detectMultiScale(gray,1.3,3)
                        
            #Gözün tespit edilmesi. Yüz ve göz çevresindeki dikdörtgen kutunun oluşturulması.
                        
            for (x,y,w,h) in faces:

                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
                        
                #Görüntünün kırpılması(Sadece yüz için).
                roi_gray = gray[x:x+w,y:y+h]
                roi_color = img[x:x+w,y:y+h]
                crop_img = img[y:y+h, x:x+w]
                        
                #Kırpılmış yüz görüntüsünün kaydedilmesi.
                cv2.imwrite('C://Users/Mustafa/Desktop/Age-Prediction/sample_cropped/'+str(image),crop_img)
        
        #Ana dizine geri döndürme.
        return os.chdir('C://Users/Mustafa/Desktop/Age-Prediction')

    
    #Classmethod un çağrılması ve görüntüyü griye dönüştüren işlevin tanımlanması.
    @classmethod
    def clor2gray(self):
        
        #Görüntülerin yeri.
        os.chdir('C://Users/Mustafa/Desktop/Age-Prediction/sample_cropped/')

        #Kırpılmış görüntülerin listesi.
        crops = os.listdir()
                        
        #Tüm görüntülerde döngü oluşturma.
        for crop in crops :
                        
            #Görüntülerin okunması.
            img = cv2.imread(crop)
                        
            #Görüntünün dönüştürülmesi.
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        
            #Dönüştürülen görüntüyü kaydetme.
            cv2.imwrite('C://Users/Mustafa/Desktop/Age-Prediction/sample_greyed'+str(crop),img)
        
        #Ana dizine geri döndürme.
        return os.chdir('C://Users/Mustafa/Desktop/Age-Prediction')


if __name__ == "__main__":

    #Yazdırma dökümanının oluşturulması.
    print(__doc__)
    
    #Nesne oluşturma.
    obj = Imagescale()
    
    #Oluşturulan nesnenin özelliklerinin çağrılması.
    obj.resizeimage()
    obj.detectfaces_crop()
    obj.clor2gray()
    
    
