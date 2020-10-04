import numpy as np
from common.funcs import *


###Optimizers###

class RMSPropGraves:    
   
    def __init__(self, lr=0.001, rho=0.95):
        
        self._lr = lr
        self._h = None
        self._rho = rho
        self._epsilon = 0.0001
        
    def update(self, params, grads_params):
        
        if self._h is None:
            #初回のみ hとmを初期化
            
            self._h = {}
            self._m = {}
            
            for key, value in params.items():
                self._h[key] = np.zeros_like(value)
                self._m[key] = np.zeros_like(value)
                
        for key in params.keys():
            #h(t) = ρ * h(t-1) + (1-ρ) * g(t)^2
            self._h[key] = self._rho * self._h[key] + (1 - self._rho) * grads_params[key] * grads_params[key]
            #m(t) = ρ * m(t-1) + (1-ρ) * g(t)
            self._m[key] = self._rho * self._m[key] + (1 - self._rho) * grads_params[key]
            #∇W(t) = -(lr * ∇W(t-1) ) / sqrt( h(t) - m(t)^2 + ε )
            params[key] -= self._lr * grads_params[key] / ( np.sqrt(self._h[key] - (self._m[key] * self._m[key]) + self._epsilon) ) 
            
class Adam:
    
    def __init__(self, lr=0.001, rho1=0.9, rho2=0.999):
        
        self._lr = lr
        self._rho1 = rho1
        self._rho2 = rho2
        self._m = None
        self._v = None
        self._epsilon = 1e-8
        self._iter_count = 0
        
    def update(self, params, grads_params):
        
        #params：更新対象のlearnableパラメーターのDictionary。Affine1個に付き「weight」「bias」というkey文字列を持つ。
        #grads_params：更新対象のlearnableパラメーターの勾配のDictionary。同様にAffine1個に付き「weight」「bias」というkey文字列を持つ。

        if self._iter_count==0 or (self._m is None or self._v is None):
            #初回のみ mとvを初期化
            #初期値は0埋め
            
            self._m = {} #勾配の指数的な移動平均（補正済）
            self._v = {} #勾配の2乗の指数的な移動平均（補正済）

            for key, value in params.items():
                self._m[key] = np.zeros_like(value)
                self._v[key] = np.zeros_like(value) 
               
        for key in params.keys():
            
            #勾配の指数的な移動平均（補正前）
            self._m[key] = self._rho1*self._m[key] + (1-self._rho1)*grads_params[key] 
            #勾配の2乗の指数的な移動平均（補正前）
            self._v[key] = self._rho2*self._v[key] + (1-self._rho2)*(grads_params[key]**2)            
            
            #勾配の指数的な移動平均mの補正値m^
            m_h = self._m[key] / ( 1 - self._rho1**(self._iter_count+1) )
            #勾配の2乗の指数的な移動平均vの補正値v^
            v_h = self._v[key] / ( 1 - self._rho2**(self._iter_count+1) )
            
            params[key] -= self._lr * m_h / (np.sqrt(v_h) + self._epsilon)
            
        self._iter_count += 1

###Optimizers　終わり###