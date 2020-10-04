# -*- coding: utf-8 -*-
import numpy as np
import copy
from common.funcs import *


###Layerの親抽象クラス###

#public abstract class Layer 
class Layer:
    
    def __init__(self, name):        
        raise TypeError("このクラスは抽象クラスです。インスタンスを作ることはできません。")
        
    def forward(self, x):
        raise NotImplementedError("forward()はオーバーライドしてください。")
        
    def backward(self, dout):
        raise NotImplementedError("backwardはオーバーライドしてください。")
        
    def update_learnable_params(self, weight_decay_lmd=0):
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、update_learnable_params()はオーバーライドしてください。")
        else:
            pass
        
    def copy_params(self):
        #パラメーターをコピーして、tupleに詰め込んで返す。
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、copy_params()はオーバーライドしてください。")
        else:
            pass
        
    def overwrite_params(self, learnable_params_tpl):
        #新しい訓練対象パラメーターのtupleを受けとって上書きする。
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、overwrite_params()はオーバーライドしてください。")
        else:
            pass
        
    def keep_temporarily_learnable_params(self):
        
        #訓練対象パラメーターを一時退避する。
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、keep_tempolary_learnable_params()はオーバーライドしてください。")
        else:
            pass
        
    def adopt_learnable_params_kept_temporarily(self):
        
        #一時退避した訓練対象パラメーターを正式採用し、使用再開する。
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、adopt_learnable_params_kept_tempolary()はオーバーライドしてください。")
        else:
            pass
    
    @property
    def name(self):
        raise NotImplementedError("nameはオーバーライドしてください。")
    
    @property
    def trainable(self):
        return False
    
    @property
    def last_loss_layer(self):
        raise NotImplementedError("last_loss_layerはオーバーライドしてください。")
    
    @property
    def optimizer(self):
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、optimizerはオーバーライドしてください。")
        else:
            return None
        
    @property
    def input_shape(self):
        raise NotImplementedError("input_shapeはオーバーライドしてください。")
        
    @property
    def output_shape(self):
        raise NotImplementedError("output_shapeはオーバーライドしてください。")
        
###Layerの親抽象クラス　終わり###


###順伝播/逆伝播Layer###

#public class Affine extends Layer 
class Affine(Layer):
    #全結合層

    def __init__(self, name, input_shape, output_shape, optimizer, init_weight_option, default_init_weight_std=0.1):
        
        self._name = name
        
        #input_shape:入力データshape。tuple。
        self._input_shape = input_shape
        #output_shape:出力データshape。tuple。
        self._output_shape = output_shape
        
        input_size = input_shape[0]
        output_size = output_shape[0]
        
        init_std = calculate_init_std_weight(input_size, output_size, init_weight_option, default_init_weight_std)
        
        self._W = init_std * np.random.randn(input_size, output_size)
        self._b = np.zeros(output_size) #biasの初期値は0で埋めるのが普通。所詮は補正項でありあまり気にしない。
        
        self._x = None
        self._original_x_shape = None
        # 重み・バイアスパラメータの微分
        self._dW = None
        self._db = None
        
        #_tempは、複数エピソードを消費する訓練中で最高の性能を示したエピソードでのパラメーターを一時退避し、
        #後にそれを正式採用するための退避領域。
        #最高の性能を示さずに性能劣化のままt訓練を終えてもインスタンス化時（init時）の性能を保証するために、init時点で一時退避してしまう。
        self._W_temp = copy.deepcopy(self._W)
        self._b_temp = copy.deepcopy(self._b)
        
        #optimizer
        self._opt = optimizer            
        
    def forward(self, x, train_flg=False):
        #順伝播
        #ｘ：入力データ
        
        #xがどのようなshapeであっても対応できるように（例えば画像形式)
        self._original_x_shape = x.shape
        x = x.reshape(x.shape[0], *self._input_shape)
        self._x = x

        out = np.dot(self._x, self._W) + self._b

        return out

    def backward(self, dout):
    
        dx = np.dot(dout, self._W.T)
        self._dW = np.dot(self._x.T, dout)
        self._db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self._original_x_shape)  #入力データxのshapeに戻す
        return dx
        
    def update_learnable_params(self, weight_decay_lmd):
    
        params = {}
        params['weight'] = self._W
        params['bias'] = self._b
    
        grads = {}
        grads['weight'] = self._dW + weight_decay_lmd*self._W
        grads['bias'] = self._db
        
        self._opt.update(params, grads)
        
    def copy_params(self):
        #パラメーターをコピーして、tupleに詰め込んで返す。
        
        copy_of_W = copy.deepcopy(self._W)
        copy_of_b = copy.deepcopy(self._b)
        
        copy_of_params_tpl = (copy_of_W, copy_of_b)
        
        return copy_of_params_tpl
    
    def overwrite_params(self, params_tpl):
        #新しいパラメーターのtupleを受けとって上書きする。
        
        #weightsの上書き
        W = params_tpl[0]
        if W.shape!=self._W.shape:
            err_msg = "weightのshapeが不正です。" + str(self._name) + " 正しいshape：" + str(self._W.shape) + " 受け取ったshape：" + str(W.shape) 
            raise ValueError(err_msg)
        
        #biasesの上書き
        b = params_tpl[1]
        if b.shape!=self._b.shape:
            err_msg = "biasのshapeが不正です。" + str(self._name) + " 正しいshape：" + str(self._b.shape) + " 受け取ったshape：" + str(b.shape) 
            raise ValueError(err_msg)
            
        self._W = W
        self._b = b
        
    def keep_temporarily_learnable_params(self):
        
        #現時点でのweightとbiasを一時退避する。
        self._W_temp = copy.deepcopy(self._W)
        self._b_temp = copy.deepcopy(self._b)
        
    def adopt_learnable_params_kept_temporarily(self):
        
        #一時退避したweightとbiasを正式採用する。
        self._W = self._W_temp
        self._b = self._b_temp
        
    def sum_weights_square(self): 
        #荷重減衰（weight_decay）用
        return np.sum(self._W**2) 
    
    def print_learnable_params(self):
        print("weight:\n", self._W)
        print("bias:\n", self._b)
        
    @property
    def name(self):
        return self._name
    
    @property
    def trainable(self):
        return True
    
    @property
    def last_loss_layer(self):
        return False
    
    @property
    def optimizer(self):
        return self._opt
    
    @property
    def input_shape(self):
        return self._input_shape
        
    @property
    def output_shape(self):
        return self._output_shape
    
###順伝播/逆伝播Layer　終わり###


###Activation Layer###

#public class ReLU extends Layer 
class ReLU(Layer):

    def __init__(self, name, input_shape):
        
        self._name = name
        self._mask_negative_on_x = None
        self._input_shape = input_shape

    def forward(self, x, train_flg=False):
        
        self._mask_negative_on_x = (x <= 0)
        out = x * np.where( self._mask_negative_on_x, 0, 1.0 )

        return out

    def backward(self, dout):
        
        dx = dout * np.where( self._mask_negative_on_x, 0, 1.0 )
        
        return dx    
    
    @property
    def name(self):
        return self._name
       
    @property
    def trainable(self):
        return False
    
    @property
    def last_loss_layer(self):
        return False
    
    @property
    def input_shape(self):
        return self._input_shape
        
    @property
    def output_shape(self):
        return self._input_shape

#public class Tanh extends Layer 
class Tanh:

    def __init__(self, name, input_shape):
        
        self._out = None
        self._name = name
        self._input_shape = input_shape

    def forward(self, x, train_flg=False):
        
        out = np.tanh(x)
        self._out = out
        
        return out

    def backward(self, dout):
        
        dx = dout * (1.0 - self._out**2)
        
        return dx    
    
    @property
    def name(self):
        return self._name
       
    @property
    def trainable(self):
        return False
    
    @property
    def last_loss_layer(self):
        return False
    
    @property
    def input_shape(self):
        return self._input_shape
        
    @property
    def output_shape(self):
        return self._input_shape
    
###Activation Layers　終わり###

###Actorの出力###

class Actor_Output():
    #Actor側の出力　正規分布の平均μまたは分散varのlog_varの「どちらか」を出力
    #class AffineとActivation（class Tanh）の2つにより構成される

    #順伝播入力は、直前中間Layerからの出力m　(N, M)　Mは直前中間Layerのニューロン数
    #順伝播出力は、out_a（muまたはlog_var）　(N, action_dim)
    
    #逆伝播入力は、d_out_a（d_muまたはd_log_var）　(N, action_dim)
    #逆伝播出力は、直前中間Layerへのd_m　(N, M)
    
    def __init__(self, name, input_shape, output_shape, optimizer):
        
        self._afn = Affine(name+"_afn", input_shape, output_shape, optimizer, init_weight_option="xavier")
        self._tanh = Tanh(name+"_tanh", output_shape)
        self._name = name
        
    def forward(self, m, train_flg=False):
        
        out_afn = self._afn.forward(m, train_flg)
        out_tanh = self._tanh.forward(out_afn, train_flg)
        out_a = out_tanh #muまたはlog_var
        
        return out_a
    
    def backward(self, d_out_a):
        
        d_out_tanh = self._tanh.backward(d_out_a)
        d_out_afn = self._afn.backward(d_out_tanh)        
        d_m = d_out_afn
        
        return d_m
        
    def update_learnable_params(self, weight_decay_lmd):
        
        self._afn.update_learnable_params(weight_decay_lmd)
        
    def copy_params(self):
        
        return self._afn.copy_params()
    
    def overwrite_params(self, params_tpl):
        
        self._afn.overwrite_params(params_tpl)
    
    def keep_temporarily_learnable_params(self):
        
        #現時点でのAffineのweightとbiasを一時退避する。
        self._afn.keep_temporarily_learnable_params()
        
    def adopt_learnable_params_kept_temporarily(self):
        
        #一時退避したAffineのweightとbiasを正式採用する。
        self._afn.adopt_learnable_params_kept_temporarily()
        
    def sum_weights_square(self): 
    
        return self._afn.sum_weights_square()    
        
    @property
    def name(self):
        return self._name
       
    @property
    def input_shape(self):
        return self._afn.input_shape
        
    @property
    def output_shape(self):
        return self._tanh.output_shape

###Actorの出力　終わり###

###損失Layers###

class Actor_Loss():
    #Actor側の損失　Lact = 1/N( Σ( Σ(-log_π(a,s|θ))*Advantage ) )
    
    ##順伝播入力は、直前Layersの出力の以下3つとGの合計4個
    ##・Actor_mu_Outputからのmu　(N, action_dim)
    ##・Actor_log_var_Outputからのlog_var　(N, action_dim)
    ##・Critic_Output（class Affine）からのVa　(N, 1)　→Advantage計算
    ##・割引報酬和G　（N, 1）　→Advantage計算
    ##　この「N」は、NNの入力のバッチ「N」（＝このlayerの入力の「N」）と同じ並びのStatesである前提（当たり前）
    ##順伝播出力は、La　(スカラー)
    
    ##逆伝播入力は、dLact　(スカラー）
    ##逆伝播出力は、直前Layersへ向けた以下2つ
    ##・Actor_mu_Output_Layerへのd_mu　(N, action_dim)
    ##・Actor_logvar_Output_Layerへのd_log_var　(N, action_dim)
    ##・×やらない！！→Critic_Output_Layerへのd_Va　(N, 1)
    
    def __init__(self, name):
        
        self._name = name
        self._La = None #このLossインスタンスからの出力値La
        self._mu = None #このLossインスタンスへの入力値mu
        self._log_var = None #このLossインスタンスへの入力値log_var
        self._action = None #このLossインスタンスへの入力値action
        self._Va = None #このLossインスタンスへの入力値V　Criticの出力から
        self._advantage = None
        self._batch_size = 0 #このLossインスタンスに順伝播してきたバッチのサイズ
        self._action_dim = 0 #actionの次元＝このLossインスタンスに順伝播してきたmuやlog_varのaxis=1
                
    def forward(self, mu, log_var, action, Va, G, softplus_to_advantage):
        #Actorの損失を算出
        #mu：Actorの出力μ　(N, action_dim)
        #log_var：Actorの出力log(var)　(N, action_dim)
        #action：取ったaction　(N, action_dim)
        #Va：Criticの出力V（で、Actorのlossに流れてきた）　(N, 1)　※Criticの出力への逆伝播はしない
        #G：割引報酬和　（N, 1）
        
        self._mu = mu
        self._log_var = log_var
        self._action = action
        self._Va = Va
        self._action_dim = mu.shape[1]
        
        #Advantage
        #(N, 1)
        self._advantage = G - Va
        
        if softplus_to_advantage==True:
            #softplusでマイナス補正する。critic側への逆伝播はしないので、softplusの微分などは考えなくて良い。
            #https://www.atmarkit.co.jp/ait/articles/2004/22/news014.html
            self._advantage = np.log(1.0 + np.exp(self._advantage))
            
        #バッチ内のデータ1件1件のlossの全件分の配列「c」
        #c = -1.0 * np.sum( -log_var - ( (action - mu)**2 / num.exp(log_var) ) , axis=1, keepdims=True) * advantage
        #右辺冒頭の -1.0 * は、損失関数にして最小化したいから。 
        #a = -log_var + ( (action -mu)**2 / num.exp(log_var) )　　のshapeは(N, action_dim)
        #b = -1.0 * np.sum( -log_var + ( (action - mu)**2 / num.exp(log_var) )  , axis=1, keepdims=True)　のshapeは(N, 1)
        #↑actionの各次元（0 ～ action_dim-1）の数値を合算している。
        #cのshapeは(N, 1)
        
        #(N, action_dim)
        a = -log_var - ( (action - mu)**2 / np.exp(log_var) )
        
        #(N, 1)
        b = -1.0 * np.sum(a , axis=1, keepdims=True) 
        
        #(N, 1)
        c = b * self._advantage
        
        self._batch_size = c.shape[0]
        #スカラー
        self._La = np.sum(c) / self._batch_size     
        
        return self._La
    
    def backward(self, d_La):
        
        #順伝播時はcのバッチ平均を取ってLaとしたので、axis=0の方向（バッチサイズの方向）にブロードキャストして、バッチサイズで割る
        #(N, 1)
        d_c = ( np.broadcast_to( d_La, (self._batch_size, 1) ) ) / self._batch_size
        
        #(N, 1)
        d_b = d_c * self._advantage #d_cもself._advantageも(N, 1)
        
        #順伝播時はaをaxis=1でsumして-1をかけたので、axis=1の方向（action_dimの方向）にブロードキャストして、-1をかける
        #(N, action_dim)
        d_a =  ( np.broadcast_to( d_b, (self._batch_size, self._action_dim) ) ) * (-1.0)
        
        #途中の数々のドラマは省略
        
        #(N, action_dim)
        d_mu = d_a * 2.0 * (self._action - self._mu) / np.exp(self._log_var)
        
        #(N, action_dim)
        d_log_var = d_a * ( ( (self._action - self._mu)**2 / np.exp(self._log_var) ) - 1.0 )        
        
        return d_mu, d_log_var

    
class Critic_Loss():
    #Critic側の損失　平均2乗和損失　L_cri = 1/2N( Σ( Σ(V-t)^2 ) )
    
    ##順伝播入力は、直前Layerの出力1個と教師信号1個の計2個
    ##・Critic_Output_LayerからのV　(N, 1)
    ##・教師信号t　(N, 1)　※実際は割引報酬和G
    ##順伝播出力は、Lc　(スカラー)
    
    ##逆伝播入力は、d_Lc　(スカラー）
    ##逆伝播出力は、直前Layerへ向けた以下1個
    ##・Critic_Output_Layerへのd_V　(N, 1)
    
    def __init__(self, name):
        
        self._name = name
        self._Lc = None #このLossインスタンスからの出力値Lc
        self._Vc = None #このLossインスタンスへの入力値V　Criticの出力から
        self._t = None #教師データ
                
    def forward(self, Vc, t):
        
        self._t = t
        self._Vc = Vc
        
        batch_size = self._Vc.shape[0]    
        self._Lc = 0.5 * np.sum((self._Vc - self._t)**2) / batch_size
        
        return self._Lc

    def backward(self, d_Lc):
        #d_Lc：逆伝播してきた勾配　（スカラー）
        
        batch_size = self._t.shape[0]
        d_self = (self._Vc - self._t) / batch_size #自計算ノード微分　(N, 1)
        d_V = d_Lc * d_self
        
        return d_V
    
    @property
    def name(self):
        return self._name


class Loss_Compounded():
    #Actorの損失LaとCriticの損失Lcの重み付き和。モデルの損失。この損失の勾配を逆伝播する。
    #L = La + 重み係数wc * Lc
        
    #initで重み係数wc（他所で設定されたハイパーパラメーター）を受け取る
    
    ##順伝播入力は、Actorの損失LaとCriticの損失Lcの2個。いずれもスカラー。
    ##・Actor_LossからのLa　(スカラー)
    ##・Critic_LossからのLc　(スカラー)
    #順伝播出力は、L　(スカラー) = La + Lc_weighted（重み係数wc * Lc）
    
    ##逆伝播入力は、d_L　(スカラーで、値は必ず「1」）
    ##逆伝播出力は、直前のActorとCriticのLoss_Layerへ向けた以下2個
    ##・Actor_Lossへのd_La　(スカラー)　加算しかないので自ノード微分は1＆d_Lが1なので必ず1*1=1のはず
    ##・Critic_Lossへのd_Lc　(スカラー)　wcでの乗算＆加算＆d_Lが1なので必ず1*1*wc=wcのはず
    
    def __init__(self, name, wc):
        
        self._name = name
        self._L =None #このLossインスタンスからの出力値　損失L
        self._wc = wc #Lcにかけ算する重み係数
        self._La = None #このLossインスタンスへの入力値　ActorのLoss　スカラー
        self._Lc = None #このLossインスタンスへの入力値　CriticのLoss　スカラー
        
    def forward(self, La, Lc):
        #ActorのLossとCriticのLossを統合する。重み係数による線形和。
        
        #Lcに重み係数をかけ算
        Lc_weighted = Lc * self._wc
        
        #LaとLc_weightedを加算
        self._L = La + Lc_weighted
        
        return self._L
    
    def backward(self, d_L=1):
        
        d_La = d_L * 1.0
        
        d_Lc = d_L * self._wc
        
        return d_La, d_Lc
    
    def copy_params(self):
        #パラメーターをコピーして、tupleに詰め込んで返す。
        
        copy_of_wc = copy.copy(self._wc)
        copy_of_params_tpl = (copy_of_wc,) #最後の「,」が無いとtupleと認識されない。
        
        return copy_of_params_tpl
    
    def overwrite_params(self, params_tpl):
        #新しいパラメーターのtupleを受けとって上書きする。
        
        #wcの上書き
        wc = params_tpl[0]        
        self._wc = wc
    
    @property
    def name(self):
        return self._name
    
    @property
    def wc(self):
        return self._wc
    
    @wc.setter
    def wc(self, wc):
        self._wc = wc
        
###損失Layers　終わり###

###正則化Layer###

#public class Dropout extends Layer 
class Dropout(Layer):
    
    def __init__(self, name, input_shape, dropout_ratio=0.5):
        
        self.dropout_ratio = dropout_ratio
        self._mask_on_x = None        
        self._name = name
        self._input_shape = input_shape

    def forward(self, x, train_flg=False):
        
        if train_flg==True:
            #np.random.rand(*x.shape)：xと同じ形状で、数値が0.0以上、1.0未満の行列を返す
            #直後にdropout_ratio(0以上1以下)との大小比較をするので、同様に0.0以上、1.0未満の乱数を返すrandでなければならない。
            self._mask_on_x = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self._mask_on_x
        else:
            #重みスケーリング推論則（weight scaling inference rule）
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self._mask_on_x
    
    @property
    def name(self):
        return self._name
    
    @property
    def trainable(self):
        return False
    
    @property
    def last_loss_layer(self):
        return False
    
    @property
    def input_shape(self):
        return self._input_shape
        
    @property
    def output_shape(self):
        return self._input_shape
    
###正則化Layer　終わり###