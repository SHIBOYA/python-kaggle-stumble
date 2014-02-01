import matplotlib.pylab as mp
import numpy as np
import pandas as pn
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import BaseNB,GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import json
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier

from nltk.stem import LancasterStemmer,SnowballStemmer 
from nltk.stem.snowball import EnglishStemmer 
from nltk import word_tokenize

#Removing Snowball makes the accuracy jump to a great extent, discard this
class Snowball(object): 
    def __init__(self): 
        self.sstem = SnowballStemmer("english", ignore_stopwords=True)
    def __call__(self, doc): 
        return [self.sstem.stem(t) for t in word_tokenize(doc)]


def main():
    traindata = list(np.array(pn.read_table("C:\Users\DAN\Documents\Python Scripts\Kaggle\\train.tsv",encoding='utf-8'))[:,2])
    testdata = list(np.array(pn.read_table("C:\Users\DAN\Documents\Python Scripts\Kaggle\\test.tsv"))[:,2])
    y = np.array(pn.read_table("C:\Users\DAN\Documents\Python Scripts\Kaggle\\train.tsv"))[:,-1]
    y = y.astype(int)
    
    X_sum = traindata + testdata
    lentrain = len(traindata)
    X_sum = transTFID(X_sum)
    X = X_sum[:lentrain]
    X_test = X_sum[lentrain:]  
    print X.shape
     
    
    kf = StratifiedKFold(y,n_folds=5)
    
#LSA with Truncated SVD, It won't show good results until we disallow stopwords and provide more word features
    tsvd = TruncatedSVD(n_components=250)
    print 'Fitting the SVD'
    tsvd.fit(X_sum)
    X_svd = tsvd.transform(X_sum)
    X_svdtrain = X_svd[:lentrain]
    X_svdtest = X_svd[lentrain:]
    print X_svd.shape
    
    #modellog = enlog(X,y,kf,X_test)
    #modellog = enlog(X_svdtrain,y,kf,X_svdtest)
   
   
   #modelforest = ranfor(X_svdtrain,y,X_svdtest,kf)
    #modelboost = grb(X,y)

    #modelknn = knn(X_svdtrain,y,X_svdtest)    
    
   # modelnaive = navb(X,y,X_test)
    modellog = log(X_svdtrain,y,X_svdtest)    
    
    #write(list((modelout+modelnaive)/2))
   # write(list(modelnaive))
    #write(list(modelforest)) 

    write(list(modellog))
    
    del (traindata,testdata,X_sum,X,X_test)
    
    #=====JSON PARSING THE COMPONENTS SEPARATELY, doesnt yield good results====
#     #traintitle,trainurl,trainbody = jsonTransform(traindata)
#     train,trainbody = jsonTransform(traindata)
#     #testtitle,testurl,testbody = jsonTransform(testdata)
#     test,testbody = jsonTransform(testdata)
#     
#     lenalltrain = len(list(train[:,2]))
#     
#     X_title  = transTFID(list(train[:,2])+list(test[:,2]))  
#     X_traintitle = X_title[:lenalltrain]
#     X_testtitle = X_title[lenalltrain:]
#    
#     
#     X_body  = transTFID(trainbody+testbody)
#     X_trainbody = X_body[:lenalltrain]
#     X_testbody = X_body[lenalltrain:]
#    
#     X_url  = transTFID(list(train[:,1])+list(test[:,1]))
#     
#     X_trainurl = X_url[:lenalltrain]
    #X_testurl = X_url[lenalltrain:]
#    mtitle = log(X_traintitle,y,X_testtitle)
#    mbody = log(X_trainbody,y,X_testbody)
#    murl = log(X_trainurl,y,X_testurl)
#    
#    
#    comb = np.hstack(mtitle,mbody,murl)
#    linearCombo(comb,y,X_test)    
#    write(list((mtitle+mbody+murl)/3))
#==============================================================================
   
  #  del (traintitle,trainurl,trainbody,testtitle,testurl,testbody)
   # del (train,test,trainbody)
          



def transTFID(data):
#For calc TFID vector, not taking stopwords improves accuracy by great extent in Logistic Regression
#Taking stopwords has a little improvement in the accuracy of random forest, we'll go with Logistic so.   
    tfv = TfidfVectorizer(min_df=2, max_features=None, strip_accents='unicode',  analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
    print "fitting pipeline and transforming for ", len(data), ' entries'
    tfv.fit(data)
    vect = tfv.transform(data)
    print vect.shape
    return vect

def log(X,y,X_test):        

    
    rd = lm.LogisticRegression(C=1, tol=0.0001,penalty='l2',dual=True, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)

    c = cross_validation.cross_val_score(rd, X, y, cv=5, scoring='roc_auc')
    print "5 Fold CV Score: for Logistic ", np.mean(c) , "+/-" , np.std(c)*2
    print c
    rd.fit(X,y)
    print 'Logist-ed'
    out = rd.predict_proba(X_test)[:,1]
    #return out
    return np.array(out)
    
#Ensemble of Logistic Regression
def enlog(X,y,kf,X_test):        

    rd = lm.LogisticRegression(C=1, tol=0.0001,penalty='l2',dual=True, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
    c = cross_validation.cross_val_score(rd, X, y, cv=5, scoring='roc_auc')
    print "5 Fold CV Score: for Logistic ", np.mean(c) , "+/-" , np.std(c)*2
    print c    
    out = np.zeros((kf.n_folds,3171))
    i = 0
    for train_index, test_index in kf:
        rd.fit(X[train_index],y[train_index])
        out[i] = rd.predict_proba(X_test)[:,1]
       # print 'ROC', roc_auc_score(y[test_index], rd.predict_proba(X[test_index])[:,1])
        i = i+1
        
    print 'Ensemble Logist-ed'
    return np.mean(out,axis=0)
    #return out[3]


#Random Forests
def ranfor(X,y,X_test,kf):
    rf = RandomForestClassifier(n_estimators=100)
    
    c = cross_validation.cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
    print "5 Fold CV Score: for Logistic ", np.mean(c) , "+/-" , np.std(c)*2
    print c 
    i = 0
#    for train_index, test_index in kf:
#        if(i==1):        
#            rf.fit(X[train_index],y[train_index])
#            #out[i] = rd.predict_proba(X_test)[:,1]
#        i = i+1
           
    rf.fit(X,y)
    print 'Random Forest-ed'
    out = rf.predict_proba(X_test)[:,1]
    return np.array(out)
 


def knn(X,y,X_test):
    kn = KNeighborsClassifier(n_neighbors=10)
    c = cross_validation.cross_val_score(kn, X, y, cv=5, scoring='roc_auc')
    print "5 Fold CV Score: for kNN ", np.mean(c) , "+/-" , np.std(c)*2
    print c    
    kn.fit(X,y)
    kn.predict_proba(X_test)

#=====Naive Bayes yield poor CV scores and so-so final scores==============
# #Naive Bayes   
# def navb(X,y,X_test):
#     nb = MultinomialNB()
#     c = cross_validation.cross_val_score(nb, X, y, cv=10, scoring='roc_auc')
#     print "5 Fold CV Score: for Random Forest ", np.mean(c) , "+/-" , np.std(c)*2
#     print c
#     nb.fit(X,y)
#     print 'Naiv-ed'
#     out = nb.predict_proba(X_test)[:,1]
#     return np.array(out)
#==============================================================================


def write(pred):
    testfile = pn.read_csv('C:\Users\DAN\Documents\Python Scripts\Kaggle\\test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = pn.DataFrame(pred, index=testfile.index, columns=['label'])
    pred_df.to_csv('C:\Users\DAN\Documents\Python Scripts\Kaggle\\stumbleforest.csv')
   
def jsonTransform(data):
    jDeco = json.JSONDecoder(encoding='utf-8')
    outtitle = []
    outbody = []
    outurl = []
    out = np.empty((len(data),3),dtype='|S1000')    
    i=0
    for train in data:    
    
        jdata = jDeco.decode(train)
        
        for jpart in jdata:
           
            if(jpart=="body"): 
                outbody.append(str(jdata[jpart]))
                s = str(jdata[jpart])
                out[i,0] = s
            elif(jpart=="url"):
                #outurl.append(str(jdata[jpart]))
                out[i,1] = str(jdata[jpart])
            elif(jpart=="title"):
                #outtitle.append(str(jdata[jpart]))
                out[i,2] = str(jdata[jpart])
                      
        i = i+1
        
    print 'Parsed json for', len(data), ' entries'
    #print len(outtitle), len(outbody), len(outurl)
    #print out.shape
    return out,outbody

# Demonstraing the use of Grid Search to effectively search for parameters best suited
#==============================================================================
#param_grid = {'C': [1, 10],'tol':[0.0001,0.001]}
# g = GridSearchCV(rd,param_grid,cv=20)
# 

# g.fit(X,y)
# 
# print("Best score: %0.3f" % g.best_score_)
# 
# print("Best parameters set:")
# best_parameters = g.best_estimator_.get_params()
# for param_name in sorted(param_grid.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
#==============================================================================
    
# For support vector machines but slow and doesnt work that well.
#==============================================================================
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100]}]
#                     
# clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5)
# clf.fit(X, y)
# 
# print("Best score: %0.3f" % clf.best_score_)
# print("Best parameters set:")
# best_parameters = clf.best_estimator_.get_params()
# for pa in tuned_parameters:
#     for paname in sorted(pa.keys()):
#         print("\t%s: %r" % (paname, best_parameters[paname]))
#==============================================================================
  

#sv = SVC(C=1,kernel='linear')



if __name__ == '__main__': 
    main()