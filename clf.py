
#Kütüphanelerin eklenmesi.

import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
#"Classmethod" un oluşturulması.
class classifier(object):

    #Classmethod un çağrılması ve veri çerçevesini yüklemek için işlevin tanımlanması.
    @classmethod
    def dataload(self,path):
        
        #Özellikler ve çıktılar için csv veri kümesini yüklenmesi.
        return pd.read_csv(path)

    #Classmethod un çağrılması ve veri çerçevesinden dedatağerleri yüklemek için işlevin tanımlanması.
    @classmethod
    def loadvalues(self,df1,df2):
        
        #Değerlerin ayıklanması.
        x = df1.iloc[:,:-1].values
        y = df2.iloc[:,:].values

        #Yüklenen değerlerin döndürülmesi.
        return x,y

    #Classmethod un çağrılması.Bölünmüş verileri eğitmek ve test etmek için işlevin tanımlanması.

    @classmethod
    def splitdata(self,x,y,size):
        
        #Veri kümesinin eğitim ve testlere ayrılması.
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = size, random_state = 0)

        #Eğitim ve test verilerinin geriye döndürülmesi.
        return X_train, X_test, y_train, y_test


    #Ölçekleme özellikleri için Classmethod çağrılması.
    @classmethod
    def scale(self,x):
        
        # Özelliklerin ölçeklendirilmesi.
        sc= StandardScaler()
        x = sc.fit_transform(x)

        #Ölçeklenmiş verilerin geri döndürülmesi.
        return x

    #Classmethod un çağrılması ve boyut küçültme için fonksiyonun tanımlanması.
    @classmethod
    def dimreduction(self,x,n):
        

        #Boyutsal küçülmenin yapılması.
        kpca = KernelPCA(n_components = n, kernel = 'rbf')
        x = kpca.fit_transform(x)

        #Verinin geri döndürülmesi.
        return x
    

    #Sınıflandırıcı tanımlamak için Classmethod çağrısının yapılması.
    @classmethod
    def clf(self,X_train,y_train):
        
        # Eğitim setine "Basit Doğrusal Regresyon" uygulanması.
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        #Geri döndürülmesi.
        return regressor

    #Veri kaydetme işlevini tanımlamak için Classmethod un çağrılması.
    @classmethod
    def save_clf(self,clf,path_with_name,_type):
        
        #Verileri boşaltmak.
        return pickle.dump(clf, open(path_with_name, _type))

    #Veri çerçevesinin yüklenmesi.

if __name__ == "__main__":

    #Yazdırma dökümanının oluşturulması.
    print(__doc__)

    #Nesne oluşturma.
    obj = classifier()

    #Yol verme.
    path1 = "C://Users/Mustafa/Desktop/Age-Prediction/sonDataset.csv"
    path2 = "C://Users/Mustafa/Desktop/Age-Prediction/sonDataset1.csv"

    
    
    df1 = obj.dataload(path1)
    df2 = obj.dataload(path2)
    
#    df3 = pd.read_csv('Train.csv',sep=';')
#    df3 = df3.drop('resim',axis=1)
#    df3 = df3.drop('bilmiyom',axis=1)
#    
#    df4 = pd.read_csv('images.csv')
#    df4 = df4.drop(columns=['Unnamed: 0'])
    
    #Yükleme değerleri.
#    x,y = df1,df2
    x,y = obj.loadvalues(df1,df2)
    #Bölünmüş veri boyutunun tanımlanması.
    size = 1/3

    #Verileri bölme.
    X_train, X_test, y_train, y_test = obj.splitdata(x,y,size)

    #Verileri ölçekleme
    X_train = obj.scale(X_train)
    X_test  = obj.scale(X_test)

    """
    #feature reduction
    X_train = obj.dimreduction(X_train,20)
    X_test  = obj.dimreduction(X_test,20)
    """

    #Sınıflandırıcının tanımlanması.
    clf = obj.clf(X_train,y_train)
    
    clf.score(X_test,y_test)
    
    path_with_name = "clf4"
    _type = "create"
    
    #Yöntemin kaydedilmesi.
    obj.save_clf(clf,path_with_name,_type)
    
    with open('clf4.pkl', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

