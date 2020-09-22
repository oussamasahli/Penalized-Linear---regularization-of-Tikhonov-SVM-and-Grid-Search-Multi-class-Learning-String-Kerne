import arftools as at
#from arftools import *
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
import math

# J'ai mis les résultats de tous mes tests dans le fichier tme3.docx (format pdf: tme3.pdf)
# J'ai commenté mes tests , comme ça si vous voulez lancer des tests en particulier,
# ça vous évitera de tous commenter à chaque fois. Il vous suffira de décommenter 
# le test qui vous intéresse. Pour chaque test je met un titre où je dis ce que je fais dans le test 


def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    s=0
    n=len(datax)
    j=0
    for row in datax:
        p=np.dot(w,row)    
        e=datay[j]-p
        j+=1
        s=s+math.pow(e,2)
    return s/n
    


def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    n_w=np.zeros(len(w))
    n=len(datax)
    
    for i in range (0,len(n_w)):
        s=0
        for j in range(0,len(datax)):
            v=np.dot(w,datax[j])
            s = s + (-datax[j][i] * ( datay[j] - v ) )
        s=s*(2/n)
        n_w[i]=s
    
    return n_w
            
        
      
    
def hinge(datax,datay,w,a=0,l=1):
    """ retourn la moyenne de l'erreur hinge """
    
    s=0
    n=len(datax)
    
    for j in range (0,len(datax)):
        p=np.dot(w,datax[j])    
        p=p*(-datay[j])
        s=s+max(0,a+p) + l*math.pow(np.linalg.norm(w),2)
        
    return s/n


def hinge_g(datax,datay,w,a=0,l=1):
    """ retourne le gradient moyen de l'erreur hinge """
    n_w=np.zeros(len(w))
    n=len(datax)

    for j in range (0,len(datax)):
        p=np.dot(w,datax[j])     
        p=p*datay[j]
        if (p<0):
            n_w+=(-datay[j]*datax[j]) + 2*l*w
            
    n_w*=1/n
   
    return n_w




class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01,biais=False):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """       
        self.biais=biais,
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        

    def fit(self,datax,datay,type_d="batch", testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        #datay = datay.reshape(-1,1)
        #N = len(datay)
        #datax = datax.reshape(N,-1)
        D = datax.shape[1]
        #self.w = np.random.random((1,D+1))
        self.errorstrain=np.zeros(self.max_iter) # calcul l'erreur moyenne à chaque ittération
        self.errorstest= np.zeros(self.max_iter)
        
        if (self.biais==(True,)):
            self.w = np.random.random(D+1) 
            # biais  sur train
            x=np.zeros((len(datax),len(datax[0])+1))
            for i in range (0,len(datax)):
                x[i][0]=1
                x[i][1:]=datax[i]
            datax=x   
            #biais sur test
            try:
                x=np.zeros((len(testx),len(testx[0])+1))
                for i in range (0,len(testx)):
                    x[i][0]=1
                    x[i][1:]=testx[i]
                testx=x
            except:
                pass #testx est None
            
                       
        else:
            self.w = np.random.random(D)
        
        
        
        if (type_d=="mini-batch"):
            di=[x for x in range(len(datax)) if x != 0 and len(datax)%x == 0]
            pas=di[np.random.randint(0,len(di))]
    
            for i in range(0,self.max_iter):
                self.errorstrain[i]=self.loss(datax,datay,self.w)
                try:
                    self.errorstest[i]=self.loss(testx,testy,self.w)
                except:
                    pass
                    
                
                for j in range(0,len(datax),pas):
                    x=datax[j:j+pas,]
                    y=datay[j:j+pas,]
                    d=self.loss_g(x,y,self.w)
                    self.w=self.w-(self.eps*d)
                    
                       
        elif (type_d=="stochastique"):
            for i in range(0,self.max_iter):
                self.errorstrain[i]=self.loss(datax,datay,self.w)
                try:
                    self.errorstest[i]=self.loss(testx,testy,self.w)
                except:
                    pass
                
                for j in range(0,len(datax)):
                    x=datax[j].reshape(1,-1)
                    y=datay[j].reshape(1,-1)
                    
                    d=self.loss_g(x,y,self.w)
                    self.w=self.w-(self.eps*d)
        else: #batch
            for i in range(0,self.max_iter):
                self.errorstrain[i]=self.loss(datax,datay,self.w)
                try:
                    self.errorstest[i]=self.loss(testx,testy,self.w)
                except:
                    pass
                
                d=self.loss_g(datax,datay,self.w)
                self.w=self.w-(self.eps*d)
            
            

    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
            
        if(self.biais==(True,)):
            x=np.zeros((len(datax),len(datax[0])+1))
            for i in range (0,len(datax)):
                x[i][0]=1
                x[i][1:]=datax[i]
               
            datax=x

        pred=[]
        for row in datax:
            p=np.dot(self.w,row)
            pred.append(np.sign(p))  # sign(<w.x>)
        return np.array(pred)   
            
            

    def score(self,datax,datay):
        
        pred=self.predict(datax)
        c=0
        for i in range(0,len(pred)):
            if(pred[i]==datay[i]):
                c+=1
        return c/len(datay)*100



def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=at.make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()
    
    
################################################################################
#Test Perceptron
################################################################################
    

#datay = datay.reshape(-1,1)
#N = len(datay)
#datax = datax.reshape(N,-1)
#D = datax.shape[1]
#w = np.random.random((1,D+1))

if __name__=="__main__":
    """ Tracer des isocourbes de l'erreur """
    plt.ion()
#    trainx,trainy =  at.gen_arti(nbex=1000,data_type=0,epsilon=1)
#    testx,testy =  at.gen_arti(nbex=1000,data_type=0,epsilon=1)
#    plt.figure()
#    plot_error(trainx,trainy,mse)
    #plt.figure()
    #plt.hist(trainx, bins='auto')
    #plt.figure()
    #plt.hist(trainy, bins='auto')
#    plt.figure()
#    plot_error(trainx,trainy,hinge)
#    perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.1)
#    perceptron.fit(trainx,trainy,"batch")
#    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
#    plt.figure()
#    at.plot_frontiere(trainx,perceptron.predict,200)
#    at.plot_data(trainx,trainy)
    
    # Je cherche à optimiser le paramètre max_iter
    
#    for i in range (10,1100,100):
#         meantrain=[]
#         meantest=[]
#         for j in range(0,10):
#             regression = Lineaire(mse,mse_g,max_iter=i,eps=0.1,biais=False)
#             regression.fit(trainx,trainy,"batch")
#             meantrain.append(regression.score(trainx,trainy))
#             meantest.append(regression.score(testx,testy))
#    print("Erreur moyenne : train %f, test %f, max_iter %d"% (np.mean(meantrain),np.mean(meantest),i))


# Affiche sur une courbe l'erreur en apprentissage
    
#perceptron = Lineaire(hinge,hinge_g,max_iter=310,eps=0.1,biais=False)
#perceptron.fit(trainx,trainy,"batch")
#x = [i for i in range(0,perceptron.max_iter)]
#y1 = perceptron.errorstrain
#plt.plot(x, y1, label="errortrain")
#plt.legend()
#plt.show()


       
#perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.1,biais=True)
#perceptron.fit(trainx,trainy,"batch")
#print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
#plt.figure()
#at.plot_frontiere(trainx,perceptron.predict,200)
#at.plot_data(trainx,trainy)

 
################################################################################
#Test Données Usps
################################################################################
    
    
datax , datay = load_usps ( "USPS/USPS_train.txt" )

datatestx , datatesty = load_usps ( "USPS/USPS_test.txt" )
#plt.figure()
#show_usps(datax[0])

def genere_Data(datax,datay,num1,num2): # récupère les données de la classe num1 et aussi les données de la classe num2
    t=0
    for i in range (0,len(datay)):
        if ( datay[i]==num1 or   datay[i]==num2):
            t+=1
    
    X=np.zeros((t,len(datax[0])))
    Y=np.zeros(t)
    
    c=0
    for i in range (0,len(datay)):   # la classe num1 devient 1, et la classe num2 devient -1
        if ( datay[i]==num1): 
            X[c]=datax[i]
            Y[c]=1
            c+=1

        elif ( datay[i]==num2 ):
            X[c]=datax[i]
            Y[c]=-1
            c+=1
    return X,Y

# classe num1 vs classe num2 , avec affichage du vecteur w , des scores , et de la courbe d'erreur en train et en test
    
#num1=1
#num2=8

#X,Y=genere_Data(datax,datay,num1,num2)
#testX,testY=genere_Data(datatestx,datatesty,num1,num2)
#perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,biais=False)
#perceptron.fit(X,Y,"batch",testX,testY)
#print("\n ",num1," vs ",num2," : \n W: \n",perceptron.w,"\n")
#print("\n ",num1," vs ",num2,"\n")
#show_usps(perceptron.w)
#plt.figure()
#plt.imshow(perceptron.w.reshape((16,16)))
#plt.figure()
#print("Erreur : train %f, test %f"% (perceptron.score(X,Y),perceptron.score(testX,testY)))
#x = [i for i in range(0,perceptron.max_iter)]
#y1 = perceptron.errorstrain
#y2 = perceptron.errorstest
#plt.plot(x, y1, label="errortrain")
#plt.plot(x, y2, label="errortest")
#plt.legend()
#plt.show()

#Courbe de l'erreur en train et en test sur les données gen_arti
    
#trainx,trainy =  at.gen_arti(nbex=1000,data_type=0,epsilon=1)
#testx,testy =  at.gen_arti(nbex=1000,data_type=0,epsilon=1)
#perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,biais=True)
#perceptron.fit(trainx,trainy,"batch",testx,testy)
#x = [i for i in range(0,perceptron.max_iter)]
#y1 = perceptron.errorstrain
#y2 = perceptron.errorstest
#plt.plot(x, y1, label="errortrain")
#plt.plot(x, y2, label="errortest")
#plt.legend()
#plt.show()



def Tranform(datay,num): # la classe num devient 1 et les autres classes deviennent -1
    Y=np.zeros(len(datay))  
    
    for i in range (0,len(datay)):   # la classe num1 devient 1, et la classe num2 devient -1
        if ( datay[i]==num):   
            Y[i]=1
        else:
            Y[i]=-1
    return Y    

# classe num vs les autres
    
#num=9
#Y=Tranform(datay,num)
#testY=Tranform(datatesty,num)
#perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,biais=False)
#perceptron.fit(datax,Y,"batch",datatestx,testY)
#show_usps(perceptron.w)
#print("\n ",num," vs les autres : \n")
#print("Erreur : train %f, test %f"% (perceptron.score(datax,Y),perceptron.score(datatestx,testY)))
#show_usps(perceptron.w)
#plt.figure()
#plt.imshow(perceptron.w.reshape((16,16)))
#plt.figure()
#x = [i for i in range(0,perceptron.max_iter)]
#y1 = perceptron.errorstrain
#y2 = perceptron.errorstest
#plt.plot(x, y1, label="errortrain")
#plt.plot(x, y2, label="errortest")
#plt.legend()
#plt.show()

    
#################################################################################
#TEST DONNEES 2D ET PROJECTION
#################################################################################
    

trainx,trainy =  at.gen_arti(nbex=1000,data_type=0,epsilon=1)
testx,testy =  at.gen_arti(nbex=1000,data_type=0,epsilon=1)

#projection polynomiale
def phi_pol(datax):
    #φ(x) = [1, x1, x2, x1^2, x2^2, x1x2]    
    X=np.zeros((len(datax),6))
    for i in range(0,len(datax)):
        x1=datax[i][0]
        x2=datax[i][1]
        
        X[i][0]=1
        X[i][1]=x1
        X[i][2]=x2
        X[i][3]=math.pow(x1,2)
        X[i][4]=math.pow(x2,2)
        X[i][5]=x1*x2
    return X

#Xtrain=phi_pol(trainx)
#Xtest=phi_pol(testx)
  
#meanTest=[]
#meanTrain=[]
#for j in range(0,10):     
#    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,biais=False)
#    perceptron.fit(Xtrain,trainy,"batch",Xtest,testy)
#    meanTest.append(perceptron.score(Xtest,testy)) 
#    meanTrain.append(perceptron.score(Xtrain,trainy)) 
   
#print("Erreur moyenne : train %f, test %f"% (np.mean(meanTrain),np.mean(meanTest)))
#x = [i for i in range(0,perceptron.max_iter)]
#y1 = perceptron.errorstrain
#y2 = perceptron.errorstest
#plt.plot(x, y1, label="errortrain")
#plt.plot(x, y2, label="errortest")
#plt.legend()
#plt.show()
    

xmin=-5
xmax=5
ymin=-5
ymax=5
step=20
x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
grid=np.c_[x.ravel(),y.ravel()]

def proj_gauss(datax,h,grid):
    X=np.zeros((len(datax),len(grid)))
    i=0
    
    for row in datax:
        x=np.zeros(len(grid))
        for j in range(0,len(grid)):
            x[j]=np.exp( - ( np.linalg.norm(row-grid[j]) ) / h )
        X[i]=x
        i+=1
    return X


#Test pour optimiser le paramètre sigma dans la formule         

#for j in np.arange(1,10,1):
#    Xtrain=proj_gauss(trainx,j,grid)
#    Xtest=proj_gauss(testx,j,grid)
        
#    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,biais=False)
#    perceptron.fit(Xtrain,trainy,"batch",Xtest,testy)
#    print("Erreur: train %f, test %f, j: %f "% (perceptron.score(Xtrain,trainy),perceptron.score(Xtest,testy),j))       
        
   
#Xtrain=proj_gauss(trainx,1.5,grid)
#Xtest=proj_gauss(testx,1.5,grid)     
#meanTest=[]
#meanTrain=[]
#for j in range(0,10):     
#    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,biais=False)
#    perceptron.fit(Xtrain,trainy,"batch",Xtest,testy)
#    meanTest.append(perceptron.score(Xtest,testy)) 
#    meanTrain.append(perceptron.score(Xtrain,trainy)) 
        
#print("Erreur moyenne : train %f, test %f"% (np.mean(meanTrain),np.mean(meanTest)))        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    



