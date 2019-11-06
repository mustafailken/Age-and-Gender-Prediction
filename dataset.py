
#Kütüphanelerin eklenmesi.

import cv2
import os

#Veri Kümesi Sınıfı nesnelerinin oluşturulması.
class Dataset(object):

    #calling init method.
    def __init__(self,l=[],f=[],t=os.listdir()):
        self.l = l
        self.f = f
        self.t = t

    #Classmethod nesnesinin oluşturulması ve özellikler veri kümesinin tanımlanması.
    @classmethod
    def createdatafeatures(self):
        
        #Ölçeklendirilmiş veri kümesine dosya yolunun verilmesi
        fd = open('C://Users/Mustafa/Desktop/Age-Prediction/egitilecek.csv','a+')
        
        #Görüntü listesinin yaratılması
        t = os.listdir()
        
        #Tüm görüntülere göz atılması ve özelliklerini ayıklama.
        for k in t:
            
            #Görüntü okuma ve gri tonlamalı görüntüye dönüştürme.
            img = cv2.imread(k,0)
            
            #Yeniden boyutlandırma
            img = cv2.resize(img,(10,10))
            
            #Görüntü matrisinin görüntü dizinine dönüştürülmesi.
            img = img.flatten().reshape(-1,1).transpose()
            
            #Dizilerin bir csv dosyasına yazılması.
            for i in img[0]:
                fd.write(str(i)+str(","))
            fd.write("\n")        

        #Dosyanın kapatılması.
        return fd.close()
    
    #Classmethod yaratılması ve çıktı veri kümesinin oluşturulması.
    @classmethod
    def createdataoutput(self):
        
        #Çıkış veri kümesi csv dosyasının konumunun sağlanması.
        fe = open("C://Users/Mustafa/Desktop/Age-Prediction/egitilmis.csv","a+")
        
        #Dizinde bulunan görüntülerin listesini saklamak.
        t = os.listdir()
        
        #Tüm görüntülerde döngü ile dolaşılması.
        for i in range(len(t)):0
            
            #Görüntü isminde sağlanan yaş hesaplama durumu ve çıktı csv dosyasına yazılması.
            fe.write(str(abs(int(t[i].split("_")[1][:4])-int(t[i].split("_")[2][:4]))))
            fe.write("\n")

        #Dosyanın kapatılması.
        return fe.close()

#"main method" un oluşturulnası
if __name__ == "__main__":
    
    #Yazdırma dökümanının oluşturulması.
    print(__doc__)
    
    #Dizini, gri tonlamalı görüntü klasörüne dönüştürme.
    os.chdir("C://Users/Mustafa/Desktop/Age-Prediction/sample_greyed")

    #Nesne oluşturulması.
    obj = Dataset()
    
    #Nesne özelliklerinin oluşturulması.
    obj.createdatafeatures()
    obj.createdataoutput()
