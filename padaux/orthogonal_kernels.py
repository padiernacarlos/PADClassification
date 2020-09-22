"""
Created on Fri Dec 13 14:24:06 2019
@author: Carlos Padierna
"""
#import numpy as np, matplotlib.pyplot as plt
from numpy import zeros, array, exp, tanh, sqrt, power
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances
from sklearn.utils.extmath import safe_sparse_dot
#from numba import njit
"""
Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)
    Y : ndarray of shape (n_samples_2, n_features)

    degree : int, default 3
    gamma : float, default None
        if None, defaults to 1.0 / n_features

    coef0 : float, default 1
"""
def build_K_Linear():
    def K_linear(X, Y=None):       
        print('**K_linear**')
        X, Y = check_pairwise_arrays(X, Y)             
        K = zeros((X.shape[0],Y.shape[0]))
        
        if X is Y: #fit-> La gramiana K es simétrica
            for i,x in enumerate(X):
                for j,z in enumerate(Y):                
                    K[i][j] = K[j][i] = x @ z
                    if j > i:
                        break
        else: #predict-> K NO es simétrica, es K<x,x_i>
            return X @ Y.T
            #for i,x in enumerate(X):
             #   for j,z in enumerate(Y):                
              #      K[i][j] = x@z
                             
        return K
    return K_linear

def my_linear(X, Y=None, dense_output=True):    
    print('**K_linear**')
    X, Y = check_pairwise_arrays(X, Y)
    #return X@Y.T
    return safe_sparse_dot(X, Y.T, dense_output=dense_output)

def build_K_sHerm(degree):
    def K_sHerm(X, Y=None):  
        print('**K_sHerm, degree=: **',str(degree))
        X, Y = check_pairwise_arrays(X, Y)             
        K = zeros((X.shape[0],Y.shape[0]))
        
        if X is Y: #fit-> La gramiana K es simétrica
            for l,x in enumerate(X):
                for m,z in enumerate(Y):
                    summ, mult = 0, 1                 
                    for i in range(len(x)):
                        summ = 1
                        for k in range(1,degree + 1, 1):
                            if x[i] !=0 and z[i] !=0:
                                summ += H(x[i],k) * H(z[i],k) / (2**(2*degree))
                        mult *= summ                    
                    K[l][m] = K[m][l] = mult
                    if m > l:
                        break
        else: #predict-> K NO es simétrica, es K<x,x_i>
            for l,x in enumerate(X):
                for m,z in enumerate(Y):
                    summ, mult = 0, 1                 
                    for i in range(len(x)):
                        summ = 1
                        for k in range(1,degree + 1, 1):
                            if x[i] !=0 and z[i] !=0:
                                summ += H(x[i],k) * H(z[i],k) / (2**(2*degree))
                        mult *= summ                    
                    K[l][m] = mult
            
        return array(K)
    return K_sHerm

# HERMITE POLYNOMIALS
# *******************************************
def H(x_i,n): 
  if(n == 0):
    return 1
  if(n == 1):
    return x_i
  return (x_i * H(x_i,n-1) - (n-1) * H(x_i, n-2))

def build_K_gegen(degree,a):
    def K_gegen(X, Y=None):  
        print('**K_gegen, degree=: **',str(degree),' alpha= ',str(a))
        X, Y = check_pairwise_arrays(X, Y)             
        K = zeros((X.shape[0],Y.shape[0]))
        
        if X is Y: #fit-> La gramiana K es simétrica
            for l,x in enumerate(X):
                for m,z in enumerate(Y):
                    summ, mult = 0, 1                 
                    for i in range(len(x)):
                        summ = 1
                        for k in range(1,degree + 1, 1):
                            if x[i] !=0 and z[i] !=0:
                                summ += G(x[i],k,a) * G(z[i],k,a) * w(x[i], z[i],a,k)
                        mult *= summ                    
                    K[l][m] = K[m][l] = mult
                    if m > l:
                        break
        else: #predict-> K NO es simétrica, es K<x,x_i>
            for l,x in enumerate(X):
                for m,z in enumerate(Y):
                    summ, mult = 0, 1                 
                    for i in range(len(x)):
                        summ = 1
                        for k in range(1,degree + 1, 1):
                            if x[i] !=0 and z[i] !=0:
                                summ += G(x[i],k,a) * G(z[i],k,a) * w(x[i], z[i],a,k)
                        mult *= summ                    
                    K[l][m] = mult
            
        return array(K)
    return K_gegen

# GEGENBAUER POLYNOMIALS
# *******************************************
# Ref: https://www.mathworks.com/help/symbolic/gegenbauerc.html#bueod6o-2
def G(x_i,n,a): 
  if(a == -0.5):  #######2020.03.09 REVISAR a==0 o a==-0.5
      return T(x_i,n)
  if(n == 0):
      return 1
  if(n == 1):
      return 2.0*a*x_i
  return (1.0 / (n+1.0)) * ( (2.0*(n+a))* x_i * G(x_i,n-1,a) - (n+2.0*a-1) * G(x_i,n-2,a) )
 
def w(x,y,a,n):
    if (a <= 0): #######2020.03.09 REVISAR a<=0 o a<=0.5
        return 1
    else:
        iNC = pochhamer(2*a+1,n) / pochhamer(1,n)
        iNC = 1E-10 if iNC == 0 else 1/(iNC**2)
        return iNC * ( power((1-x*x)*(1-y*y),a) + 0.1) / (n+1) #######2020.03.09 REVISAR a o a-0.5

def pochhamer(x,k):
    if k==0:
        return 1.0
    if k < 0:
        return 0.0
    if x == 0:
        return 0.0
    aux = 1.0
    for i in range(0,k):
        aux *= (x+i)
    return aux

def gegenbauerc(x, n, a):

    first_value = 1.0
    second_value = 2.0 * a * x

    if n == 0:
        return first_value
    elif n == 1:
        return second_value
    else:
        result = 0.0

        for i in range(2, n + 1):
            result = 2.0 * x * (i + a - 1.0) * second_value - (
                (i + 2.0 * a - 2.0) * first_value
            )
            result /= i

            first_value = second_value
            second_value = result
        return result

def build_K_cheb(degree):
    def K_cheb(X, Y=None):  
        #print('**K_sHerm, degree=: **',str(degree))
        X, Y = check_pairwise_arrays(X, Y)             
        K = zeros((X.shape[0],Y.shape[0]))
        
        if X is Y: #fit-> La gramiana K es simétrica
            for l,x in enumerate(X):
                for m,z in enumerate(Y):
                    summ, mult = 0, 1                 
                    for i in range(len(x)):
                        summ = 1
                        for k in range(1,degree + 1, 1):
                            if x[i] !=0 and z[i] !=0:
                                summ += T(x[i],k) * T(z[i],k)
                        mult *= summ / sqrt(1.0001-x[i]*z[i])                    
                    K[l][m] = K[m][l] = mult
                    if m > l:
                        break
        else: #predict-> K NO es simétrica, es K<x,x_i>
            for l,x in enumerate(X):
                for m,z in enumerate(Y):
                    summ, mult = 0, 1                 
                    for i in range(len(x)):
                        summ = 1
                        for k in range(1,degree + 1, 1):
                            if x[i] !=0 and z[i] !=0:
                                summ += T(x[i],k) * T(z[i],k)
                        mult *= summ / sqrt(1.0001-x[i]*z[i])         
                    K[l][m] = mult
        return array(K)
    return K_cheb

# CHEBYSHEV POLYNOMIALS
# *******************************************
def T(x_i,n): 
  if(n == 0):
    return 1
  if(n == 1):
    return x_i
  return (2 * x_i * T(x_i,n-1) - T(x_i, n-2))

def build_K_rbf(gamma):
    def my_rbf(X, Y=None):
        """ K(x, y) = exp(-gamma ||x-y||^2)
        Returns kernel_matrix : array of shape (n_samples_X, n_samples_Y) """
        X, Y = check_pairwise_arrays(X, Y)
        
        #if gamma is None:
        #    gamma = 1.0 / X.shape[1]
    
        K = euclidean_distances(X, Y, squared=True)
        K *= -gamma
        exp(K, K)  # exponentiate K in-place
        return K
    return my_rbf


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """ K(X, Y) = (gamma <X, Y> + coef0)^degree   
    Returns Gram matrix : array of shape (n_samples_1, n_samples_2) """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    K **= degree
    return K

def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    """ K(X, Y) = tanh(gamma <X, Y> + coef0)
    Returns Gram matrix : array of shape (n_samples_1, n_samples_2)"""
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    tanh(K, K)  # compute tanh in-place
    return K

#def test_K_sHerm():
    #"""MUESTRA POLINOMIOS DE S-HERMITE PARA VALIDAR LA H(x_i,n) ESCALADA"""
    #plt.figure()
    #t = np.arange(-1,1.1,.1) #Rango de prueba
    #for i in range(1,6):
        #plt.plot(t, H(t,i)*2**(-i), label = 'Grado '+str(i))
    #plt.legend()
    #plt.title("Polinomios Ortogonales de s-Hermite")