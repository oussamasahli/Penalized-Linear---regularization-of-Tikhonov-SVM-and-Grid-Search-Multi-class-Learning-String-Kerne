# -*- coding: utf-8 -*-

# ICI j'ai représenté chaque partie du tme par des blocs de test. Décommentez la
# partie qui vous intéresse, comme ça ça vous évite de tous commenter dès le début.
# TOUS mes test et mes explications sont dans le fichier RAPPORT_SAHLI_OUSSAMA_TME4.pdf

#from arftools import *
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
import math
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import tme3Modifier as tme3
import arftools as at
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import scipy


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

datax , datay = load_usps ( "USPS/USPS_train.txt" )
datatestx , datatesty = load_usps ( "USPS/USPS_test.txt" )

trainx,trainy =  at.gen_arti(nbex=1000,data_type=0,epsilon=1)
testx,testy =  at.gen_arti(nbex=1000,data_type=0,epsilon=1)

###############################################################################
# TEST AVEC SKLEARN
###############################################################################

# Perceptron
#num1=1
#num2=8

#X,Y=tme3.genere_Data(datax,datay,num1,num2)
#testX,testY=tme3.genere_Data(datatestx,datatesty,num1,num2)
#perceptron = Perceptron(tol=1e-3, random_state=0)
#perceptron = tme3.Lineaire(tme3.hinge,tme3.hinge_g,max_iter=1000,eps=0.1,biais=False)
#perceptron.fit(X,Y)

#accuracytrain=[]    
#accuracytest=[]
#for i in range (0,10):
#    accuracytrain.append(perceptron.score(X,Y))
#    accuracytest.append(perceptron.score(testX,testY))
#print("Erreur moyenne : train %f, test %f"% (np.mean(accuracytrain),np.mean(accuracytest)))
#print("\n\n")
#p = perceptron.predict(testX) 
#print(classification_report(testY, p)) 

# KNN 
#knn = KNeighborsClassifier(n_neighbors=5)
#knn.fit(X, Y)

#accuracytrain=[]    
#accuracytest=[]
#for i in range (0,10):
#    accuracytrain.append(knn.score(X,Y))
#    accuracytest.append(knn.score(testX,testY))
#print("Erreur moyenne : train %f, test %f"% (np.mean(accuracytrain)*100,np.mean(accuracytest)*100))
#print("\n\n")
#p = knn.predict(testX) 
#print(classification_report(testY, p)) 


# DECISION TREE
#num1=1
#num2=8
#X,Y=tme3.genere_Data(datax,datay,num1,num2)
#testX,testY=tme3.genere_Data(datatestx,datatesty,num1,num2)
#tree = DecisionTreeClassifier(random_state=0)
#tree.fit(X, Y)

#accuracytrain=[]    
#accuracytest=[]
#for i in range (0,10):
#    accuracytrain.append(tree.score(X,Y))
#    accuracytest.append(tree.score(testX,testY))
#print("Erreur moyenne : train %f, test %f"% (np.mean(accuracytrain)*100,np.mean(accuracytest)*100))
#print("\n\n")
#p = tree.predict(testX) 
#print(classification_report(testY, p)) 
                            

###############################################################################
# TEST TIKHONOV
###############################################################################

#num1=1
#num2=8

#X,Y=tme3.genere_Data(datax,datay,num1,num2)
#testX,testY=tme3.genere_Data(datatestx,datatesty,num1,num2)
#perceptron = tme3.Lineaire(tme3.hinge,tme3.hinge_g,max_iter=1000,eps=0.1,biais=False)
#perceptron.fit(X,Y)
#svm = SVC(probability=True,gamma="scale",kernel="linear")
#svm.fit(X, Y)

#accuracytrain=[]    
#accuracytest=[]
#for i in range (0,10):
#    accuracytrain.append(svm.score(X,Y)*100)
#    accuracytest.append(svm.score(testX,testY)*100)
#print("Erreur moyenne : train %f, test %f"% (np.mean(accuracytrain),np.mean(accuracytest)))
#print("\n\n")
#p = svm.predict(testX) 
#print(classification_report(testY, p)) 


###############################################################################
# TEST SKLEARN SVM ET GRID SEARCH
###############################################################################

def plot_frontiere_proba(data,f,step =20):
    grid,x,y=at.make_grid( data=data, step=step )
    plt.contourf( x,y,f( grid ).reshape( x.shape ) , 255 )   
 

#svm = SVC(probability=True,gamma=0.1,kernel="rbf",C=0.1)
#svm.fit(X, Y)
#p = svm.predict(testx) 
#print(classification_report(testy, p)) 

#print("\nscores paramètre par défault :")
#accuracytrain=[]
#accuracytest=[]
#for i in range (0,10):
#    accuracytrain.append(svm.score(X,Y)*100)
#    accuracytest.append(svm.score(testX,testY)*100)
#print("Erreur moyenne : train %f, test %f"% (np.mean(accuracytrain),np.mean(accuracytest)))

#print(l/np.linalg.norm(svm.coef_))
#print(l[np.where(l==1)])

#l=svm.decision_function(X)
#v=0
#for r in l:
#    if ( math.fabs(r)<1.01 and math.fabs(r)>0.99 ):
#        v+=1
#print("\nnombre de vecteur de supports: ",v,"\n")

#plt.ion()
#plt.figure()
#↓plot_frontiere_proba( trainx , lambda x : svm.predict_proba(x)[:,0], step=50)
#at.plot_data(trainx,trainy)
num1=1
num2=8
X,Y=tme3.genere_Data(datax,datay,num1,num2)
testX,testY=tme3.genere_Data(datatestx,datatesty,num1,num2)

#clf = SVC(probability=True,gamma='scale')
#clf.fit(X, Y)   
#parameters = [{'kernel': ['rbf','linear','poly'],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'C': [0.1, 1, 10, 100, 1000]},
#             ]

#clf = GridSearchCV( SVC(probability=True), parameters,refit = True,cv=5)
#clf.fit(X,Y)
    
#Pour des valeurs aléatoires
#distributions = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
#  'kernel': ['rbf','linear','poly'], 'class_weight':['balanced', None]}

#clf = RandomizedSearchCV(SVC(probability=True), distributions, random_state=0,cv=5) 
#clf.fit(X,Y) 
#print("\nBest Parameters:\n")
#print(clf.best_params_)
#print("\nbest Estimator: \n")
#print(clf.best_estimator_)
#p = clf.predict(testX) 
#print(classification_report(testY, p)) 

#print("\n best_parameters: ",clf.best_params_,"\n")
#svm=clf.best_estimator_
#svm.fit(trainx,trainy)
#print("Erreur: train  %f  , test  %f",(svm.score(trainx,trainy),svm.score(testx,testy)))
    
   
#accuracytrain=[]
#accuracytest=[]
#for i in range (0,10):
#    accuracytrain.append(clf.score(X,Y)*100)
#    accuracytest.append(clf.score(testX,testY)*100)
#print("\nErreur moyenne : train %f, test %f"% (np.mean(accuracytrain),np.mean(accuracytest)))


#plt.figure()
#plot_frontiere_proba( trainx , lambda x : clf.predict_proba(x)[:,0], step=50)
#at.plot_data(trainx,trainy)
    
#scores=["accuracy","recall","f1"]
#for score in scores:
#    clf = GridSearchCV( SVC(probability=True,gamma='scale'), parameters,cv=5, scoring=score)
#    clf.fit(trainx,trainy)
#    print("\nscore:",score)
#    print("Erreur: train  %f  , test  %f"% (clf.score(trainx,trainy)*100,clf.score(testx,testy)*100))
#    print("best parameters: ",clf.best_params_)
    

##################################################################################
# TEST APPRENTISSAGE MULTICLASSE
################################################################################

# ONE VS ONE

#Ici j'utilise les données multiclasses
print("ovo:\n")
ovo=OneVsOneClassifier(SVC(C=10,gamma=0.01,kernel="rbf"))
ovo.fit(datax,datay)
p = ovo.predict(datatestx) 
print(classification_report(datatesty, p)) 

accuracytrain=[]
accuracytest=[]
for i in range (0,10):
    accuracytrain.append(ovo.score(datax,datay)*100)
    accuracytest.append(ovo.score(datatestx,datatesty)*100)
print("Erreur moyenne  : train %f, test %f"% (np.mean(accuracytrain),np.mean(accuracytest)))
s
#ONE VS REST
print("ovr:\n")
ovr = OneVsRestClassifier(SVC(C=10,gamma=0.01,kernel="rbf"))
ovr.fit(datax, datay)
p=ovr.predict(datatestx)
print(classification_report(datatesty, p)) 

accuracytrain=[]
accuracytest=[]
for i in range (0,10):
    accuracytrain.append(ovr.score(datax,datay)*100)
    accuracytest.append(ovr.score(datatestx,datatesty)*100)
print("Erreur moyenne  : train %f, test %f"% (np.mean(accuracytrain),np.mean(accuracytest)))

################################################################################
# TEST STRING KERNEL
################################################################################

#A="robuste"
#B="routage"

def Kernel_String(A,B,l=0.5):
    dA={}
    for i in range (0,len(A)-1):
        for j in range (i+1,len(A)):
            dA[A[i]+A[j]]=math.pow(l,j-i+1)
            
    dB={}
    for i in range (0,len(B)-1):
        for j in range (i+1,len(B)):
            dB[B[i]+B[j]]=math.pow(l,j-i+1)
    
    d=np.unique(list(dA.keys())+list(dB.keys()))
    
    s=0
    for c in d:
       v1=0
       v2=0
       if (c in dA):
           v1=dA[c]
       if (c in dB):
           v2=dB[c]
       s=s+(v1*v2)
    return s
     

# DATASET
    
df = pd.read_csv('train/train.csv')

import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatiser = WordNetLemmatizer()
def text_process(tex):
    nopunct=[char for char in tex if char not in string.punctuation]
    nopunct=''.join(nopunct)
    a=''
    i=0
    for i in range(len(nopunct.split())):
        b=lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a=a+b+' '
    return [word for word in a.split() if word.lower() not 
            in stopwords.words('english')]



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
y = df['author']
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

X = list(df['text'])

X_train, X_test, y_train, y_test = train_test_split(X, y
                                  ,test_size=0.2, random_state=1234)
#X_train=X_train[:50]
#y_train=y_train[:50]
#X_test=X_test[:50]
#y_test=y_test[:50]

#for l in np.arange(0.1,1,0.1):
#    print("\nlambda=",l,"\n")
#    mat=np.zeros((len(X_train),len(X_train)))
#    for i in range(0,len(X_train)):
#        x=np.zeros(len(X_train))
#        cpt=0
#        for j in range(0,len(X_train)):
#            x[cpt]=Kernel_String(X_train[i],X_train[j],l)
#            cpt+=1
#        mat[i]=x        
                
#    mat_test=np.zeros((len(X_test),len(X_test)))
#    for i in range(0,len(X_test)):
#        x=np.zeros(len(X_test))
#        cpt=0
#        for j in range(0,len(X_test)):
#            x[cpt]=Kernel_String(X_test[i],X_test[j],l)
#            cpt+=1
#        mat_test[i]=x   
    
#    clf = SVC(kernel='precomputed')
#    clf.fit(mat, y_train)  
#    p=clf.predict(mat_test)
#    print(classification_report(y_test, p)) 
    
#plt.figure()    
#plt.imshow(mat) 
#plt.colorbar()
#plt.figure()    
#plt.imshow(mat_test) 
#plt.colorbar()
    

