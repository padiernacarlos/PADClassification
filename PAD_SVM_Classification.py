#This program evaluates the SVMs trained for PAD classification as described in the paper:
#Classification Method of Peripheral Arterial Disease in Patients with type 2 Diabetes Mellitus by Infrared Thermography and Machine Learning
#Submitted to the Journal of Infrared Physics and Technology
#
# Technical Note: When the following error arises
# ....BrokenProcessPool: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable...
# The IPython terminal should be restarted. It is an issue with the Spider environment when using the joblib for Parallel

import numpy as np, pandas as pd, multiprocessing, math
from time import time
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed

#This functions are the customized orthogonal polynomial kernel functions
from padaux.orthogonal_kernels import build_K_gegen, build_K_sHerm, build_K_Linear, build_K_rbf


# loading best params
# **** DON'T FORGET REPLACE datapath  BY YOUR PATH ***
datapath = 'C:/Users/TUF-PC8/OneDrive - Universidad de Guanajuato/Manuscrito Infrarrojo-Pie Diabético/SVMs/sourceCodes/'
path  = datapath+'PAD_best_params.xlsx'
cs    = pd.read_excel(path,sheet_name='Cs')
intrs = pd.read_excel(path,sheet_name='Intrinsics')
ns    = pd.read_excel(path,sheet_name='Ns')

# loading dataset and kernel
k       = 5 # folds
do_n_trials = True
dataset = 'PAD (scaled dataset)'
kernel  = 'GEGEN'
dt      = pd.read_csv(datapath+dataset+'.csv',header=None)
X,y     = (dt.iloc[:,0:-1]) , dt.iloc[:,-1]
my_C    = np.array(cs.loc[(cs['Dataset'] == dataset)][kernel])[0]
dicKs   = {
    'LINEAR': build_K_Linear(),
    'RBF'   : build_K_rbf(float(intrs.loc[intrs['Dataset'] == dataset]['RBF'])),
    's-HERM': build_K_sHerm(int(ns.loc[ns['Dataset'] == dataset]['s-HERM'])),
    'GEGEN' : build_K_gegen(int(ns.loc[ns['Dataset'] == dataset]['GEGEN']),(-0.+
                        float(intrs.loc[intrs['Dataset'] == dataset]['GEGEN']))) # -0.5 para ajustar desplazamiento de alpha
          }

my_kernel = dicKs[kernel]
alphaTemp = float(intrs.loc[intrs['Dataset'] == dataset]['GEGEN'])
nTemp = int(ns.loc[ns['Dataset'] == dataset]['GEGEN'])
gammaTemp = float(intrs.loc[intrs['Dataset'] == dataset]['RBF'])
print("C: ",str(my_C)," alpha: ",str(alphaTemp)," n: ",str(nTemp)," gamma: ",str(gammaTemp))

start_time = time()
def validate_kernel(X,y,k,my_kernel):
    """ Stratified k-fold cross validation """   
    
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    count, cva, train_accuracy = 0,0,0
    sensa, speca, nmcca, psva = 0,0,0,0
    accs = []

    for train_index, test_index in skf.split(X, y):
        count = count + 1
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        # Building and testing the SVM
        clf    = SVC(kernel=my_kernel, C=my_C)
        model  = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)            
            
        # computing accuracy        
        train_accuracy = np.mean(y_pred.ravel() == y_test.ravel()) * 100    
        accs.append(train_accuracy)        
        cva = cva + train_accuracy
        
        tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()
        print(tn,fp,fn,tp)        
        A = (tp + fp) if (tp + fp) != 0 else 1E-5
        B = (tp + fn) if (tp + fn) != 0 else 1E-5
        C = (tn + fp) if (tn + fp) != 0 else 1E-5
        D = (tn + fn) if (tn + fn) != 0 else 1E-5
        sens = tp/B
        spec = tn/C        
        s  = (math.sqrt(A*B*C*D))
        mcc= ((tp*tn)-(fp*fn))/s
        nmcc = 0.5*(mcc+1)
        
        sensa = sensa + sens
        speca = speca + spec
        nmcca = nmcca + nmcc
        nsvs = model.n_support_
        print("#td: ", str(X_train.shape), "total =",len(X_train))
        print("#SV: ", str(nsvs ),"total =",str(nsvs[0]+nsvs[1]))
        psva = psva + ( (nsvs[0]+nsvs[1]) / len(X_train))
    
        
    print("sensitivity", str(sensa/k))
    print("specificity", str(speca/k))              
    print("nmcc", str(nmcca/k))
    print("psva", str(psva/k))
    print("cva", str(cva/k))
        
    cva = cva / k
    accs.append(psva/k)
    return accs
    
# Paraleliza n_ensayos de validación cruzada
if do_n_trials:      
    num_cores = multiprocessing.cpu_count()     
    r=Parallel(n_jobs=num_cores,verbose=20)(delayed(validate_kernel)(X,y,k,my_kernel) for i in range(1000))
    elapsed_time = time()-start_time   
    s = "{:.2f}".format(elapsed_time)
    R= np.array(r)
    stats = [R[:,5].mean(), R[:,5].std(), s]
else:
    R=np.array(validate_kernel(X,y,k,my_kernel))        
    elapsed_time = time()-start_time   
    s = "{:.2f}".format(elapsed_time)
    stats = [R[:5].mean(), R[:5].std(), s]

pd.DataFrame(stats)

print('***************************************************************')
print("Tardó: "+s+" segundos")
print("C: ",str(my_C)," alpha: ",str(alphaTemp)," n: ",str(nTemp)," gamma: ",str(gammaTemp))