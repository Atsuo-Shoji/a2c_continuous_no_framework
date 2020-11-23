# -*- coding: utf-8 -*-
import numpy as np
from collections import deque, OrderedDict
from datetime import datetime
from common.funcs import *
from common.layers import *
from common.optimizers import *

class Planner_share():
    
    def __init__(self, name, env, state_dim, action_dim, wc_critic_loss=1.0):
        
        #name：このPlannerインスタンスの名前
        #env：このPlannerインスタンスが対象とする環境
        #state_dim：このPlannerインスタンスが認識する状態の要素数
        #action_dim：このPlannerインスタンスが認識する行動の要素数
        #wc_critic_loss：モデルの損失はActorの損失とCriticの損失の重み付き線形和であり、そのCriticの損失の重み係数
        
        self._name = name
        self._env = env
        self._state_dim = state_dim
        self._action_dim = action_dim
                
        ###Layersの定義###

        ##ActorとCriticの共有Layersの定義##
        
        self._layers_shared = OrderedDict()
        
        output_dim_afn1 = self._state_dim*10
        output_dim_afn3 = self._action_dim*10
        output_dim_afn2 = np.ceil( np.sqrt(output_dim_afn1*output_dim_afn3) ).astype(np.int)
        
        #①Affine 「afn1」　instance of Affine
        opt_afn1 = Adam(lr=0.001, rho1=0.9, rho2=0.999)        
        afn1 = Affine(name="afn1", input_shape=(state_dim,), output_shape=(output_dim_afn1,), optimizer=opt_afn1, 
                        init_weight_option="xavier")
        self._layers_shared[afn1.name] = afn1
        prev_layer = afn1

        #②Tanh　「tanh_afn1」　instance of Tanh
        tanh_afn1 = Tanh(name="tanh_afn1", input_shape=prev_layer.output_shape)
        self._layers_shared[tanh_afn1.name] = tanh_afn1
        prev_layer = tanh_afn1

        #③Affine 「afn2」　instance of Affine
        opt_afn2 = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        afn2 = Affine(name="afn2", input_shape=prev_layer.output_shape, output_shape=(output_dim_afn2,), optimizer=opt_afn2, 
                        init_weight_option="xavier")
        self._layers_shared[afn2.name] = afn2
        prev_layer = afn2

        #④Tanh　「tanh_afn2」　instance of Tanh
        tanh_afn2 = Tanh(name="tanh_afn2", input_shape=prev_layer.output_shape)
        self._layers_shared[tanh_afn2.name] = tanh_afn2
        prev_layer = tanh_afn2
        
        #⑤Affine 「afn3」　instance of Affine
        opt_afn3 = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        afn3 = Affine(name="afn3", input_shape=prev_layer.output_shape, output_shape=(output_dim_afn3,), optimizer=opt_afn3, 
                        init_weight_option="xavier")
        self._layers_shared[afn3.name] = afn3
        prev_layer = afn3

        #⑥Tanh　「tanh_afn3」　instance of Tanh
        tanh_afn3 = Tanh(name="tanh_afn3", input_shape=prev_layer.output_shape)
        self._layers_shared[tanh_afn3.name] = tanh_afn3
        
        last_layer_of_shared = tanh_afn3
        
        ##ActorとCriticの共有Layersの定義　終わり##
        
        ##ActorとCritic　出力のLayersの定義##
        
        #Actorの出力layersは2個で、Criticの出力layerは1個
        #これら3個の出力layersは、「並列」であることに注意
        
        #Actorの出力は2個　正規分布の平均μを出力するLayersと、分散varのlog(var)を出力するLayers
        #Actorの出力は1個に付きAffineとTanhの2layers
        
        #Actor　平均μの出力のlayers　中身はAffine1個とTanh1個
        opt_afn_actor_mu = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        self._out_actor_mu = Actor_Output(name="out_actor_mu", input_shape=last_layer_of_shared.output_shape, 
                                          output_shape=(action_dim,), optimizer=opt_afn_actor_mu)
        
        #Actor　分散varのlog(var)の出力のlayers　中身はAffine1個とTanh1個
        opt_afn_actor_log_var = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        self._out_actor_log_var = Actor_Output(name="out_actor_log_var", input_shape=last_layer_of_shared.output_shape, 
                                               output_shape=(action_dim,), optimizer=opt_afn_actor_log_var)
                
        #Critic　価値関数Vの出力のlayer　活性化関数は無い（linear）のでAffine1個だけ
        opt_afn_critic = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        self._out_critic = Affine(name="out_critic", input_shape=last_layer_of_shared.output_shape, 
                                  output_shape=(1,), optimizer=opt_afn_critic, init_weight_option="xavier")
        
        ##ActorとCritic　出力のLayersの定義　終わり##
        
        ##ActorとCritic　Lossの定義##
        
        #ActorとCriticの各々のloss1個ずつで合計2個だが、これらは「並列」であることに注意
        
        #ActorのLoss
        self._loss_actor = Actor_Loss("loss_actor")
        
        #criticのLoss
        self._loss_critic = Critic_Loss("loss_critic")
        
        ##ActorとCritic　Lossの定義　終わり##
        
        ##モデル全体のLossの定義##
        
        self._loss = Loss_Compounded("loss_compounded", wc_critic_loss)
        
        ##モデル全体のLossの定義　終わり##
        
        ###Layersの定義　終わり###
    
    def predict_best_action(self, a_state):
        #最適な行動を推測
        
        if a_state.ndim==1:
            a_state = a_state.reshape(1, self._state_dim) 
        elif a_state.shape[0]!=1:
            raise ValueException("a_stateは1件だけにしてください。1件のstateについてbest actionを推測します。")            
        
        #Actorから正規分布の平均μと分散varのlog_varを得る。
        #どちらも(1, action_dim) 
        mu, log_var, _ = self._forward_output_layers(a_state, train_flg=False)
        
        #分散var
        #(1, action_dim) 
        var = np.exp(log_var)
        
        #muとvarからバッチ数のaxis=0を削除し、(action_dim,)にする
        mu = mu[0]
        var = var[0]
        
        #平均μと分散varの正規分布からサンプリング
        best_action = np.random.normal(loc=mu, scale=np.sqrt(var))
        
        return best_action
        
    def _forward_layers_shared(self, state, train_flg):
        #共有部分のみの順伝播
        
        if state.ndim==1:
            state = state.reshape(1, self._state_dim)  
           
        x = state
        for layer in self._layers_shared.values():
            x = layer.forward(x, train_flg)            
        
        m = x
        
        return m
        
    def _backward_layers_shared(self, d_m):
        #共有部分の誤差逆伝播（NN最初までの誤差逆伝播）
        
        layers_shared = list(self._layers_shared.values())
        layers_shared.reverse()
        d_out = d_m
        for layer in layers_shared:
            d_out = layer.backward(d_out)
        
        d_state = d_out
        
        return d_state        
        
    def _forward_output_layers(self, state, train_flg):
        #共有部分 + Actorの出力とCriticの出力
        #分岐以外の、勾配に影響を与える演算はしないこと
        
        ##共有部分の出力##
        
        #出力値を「m」とする
        #(N, M) 「M」は共有最終layerのAffineのニューロン数
        m = self._forward_layers_shared(state, train_flg)
        
        ##共有部分の出力　終わり##
        
        ##Actorの出力とCriticの出力##
        
        #3つの出力layersに共有部分の出力を渡す。
        #共有部分の出力が3分岐する。
        
        #Actorのμ出力layersからの出力 (N, action_dim)
        mu = self._out_actor_mu.forward(m, train_flg)
        
        #Actorのlog(var)出力layersからの出力 (N, action_dim)
        log_var = self._out_actor_log_var.forward(m, train_flg)
        
        #CriticのV出力layerからの出力 (N, 1)
        V = self._out_critic.forward(m, train_flg)
        
        ##Actorの出力とCriticの出力　終わり##
        
        return mu, log_var, V
    
    def _backward_output_layers(self, d_mu, d_log_var, d_V):
        #Actorの出力とCriticの出力からNN最初までの誤差逆伝播
        
        ##Actorの出力とCriticの出力の逆伝播##
        
        #Actorのμ出力部分の逆伝播
        #(N, M) 「M」は共有最終layerのAffineのニューロン数
        d_m_mu = self._out_actor_mu.backward(d_mu)
        
        #Actorのlog(var)出力部分の逆伝播
        #(N, M) 「M」は共有最終layerのAffineのニューロン数
        d_m_log_var = self._out_actor_log_var.backward(d_log_var)
        
        #CriticのV出力部分の逆伝播
        #(N, M) 「M」は共有最終layerのAffineのニューロン数
        d_m_V = self._out_critic.backward(d_V)
        
        #forwardで共有部分最終layerの出力mを
        #Actorのμ出力、Actorのlog(var)出力、CriticのV出力
        #に3分岐したので、これらから逆伝播してきた勾配を合算する。
        d_m = d_m_mu + d_m_log_var + d_m_V
        
        ##Actorの出力とCriticの出力の逆伝播　終わり##
        
        ##共有部分（NN最初まで）の逆伝播##
        #(N, state_dim)
        d_state = self._backward_layers_shared(d_m)
        ##共有部分（NN最初まで）の逆伝播　終わり##
        
        return d_state        
    
    def _forward_losses(self, state, action, G, softplus_to_advantage, weight_decay_lmd):
        #共有部分 + Actorの出力とCriticの出力 + ActorのlossとCriticのloss + 全体のloss
        #action, Gのもとになったstateは、引数stateと同じ並びであること
        #分岐以外の、勾配に影響を与える演算はしないこと
        
        ##Actorの出力、Criticの出力##
        
        #Actorの出力μ、Actorの出力log(var)、Criticの出力V
        #(N, action_dim), (N, action_dim), (N, 1)
        mu, log_var, V = self._forward_output_layers(state, train_flg=True)
        
        ##Actorの出力、Criticの出力　終わり##
        
        ##Actorのloss、Criticのloss##
        
        #以下は、Criticの出力VがActorのlossとCriticのlossへと2分岐している。
        #が、backward時は、Actorのlossからの逆伝播はさせない。
        
        #Actorのloss
        La = self._loss_actor.forward(mu, log_var, action, V, G, softplus_to_advantage)
            
        #Criticのloss
        Lc = self._loss_critic.forward(V, G)  #Gは教師信号として使用される
        
        ##Actorのloss、Criticのloss　終わり##
        
        ##モデル全体のloss##
        L = self._loss.forward(La, Lc)
        ##モデル全体のloss　終わり##
        
        #荷重減衰
        if weight_decay_lmd > 0:
            sum_all_weights_square = self._sum_all_weights_square()
            L = L + 0.5*weight_decay_lmd*sum_all_weights_square
        
        return L, La, Lc
    
    def _backward_losses(self, d_L=1):
        #全体のlossからNNの最初までの一気通貫の誤差逆伝播
        
        ##モデル全体のloss　逆伝播##
        #どちらもスカラー
        d_La,  d_Lc = self._loss.backward(d_L)
        ##モデル全体のloss　逆伝播　終わり##
        
        ##Actorのloss、Criticのloss　逆伝播##
        
        #Actorのloss　逆伝播
        #注意！Criticの出力layerへの逆伝播は行わない。よって、以下「d_Va」という3つ目の戻り値は無い。
        #どちらも(N, action_dim)
        d_mu, d_log_var = self._loss_actor.backward(d_La)
        
        #Criticのloss　逆伝播
        #注意！Criticの出力layerへのActorのlossの逆伝播は行わないので、順伝播時のVの（ActorのlossとCriticのlossへの）2分岐
        #による勾配の合算は行わない（d_V = d_Va + d_Vc はしない、ということ）。
        #(N, 1)
        d_V = self._loss_critic.backward(d_Lc)
        
        ##Actorのloss、Criticのloss　逆伝播　終わり##
        
        ##Actor、Criticの出力Layersと共有部分の逆伝播　NNの最初まで##
        d_states = self._backward_output_layers(d_mu, d_log_var, d_V)
        ##Actor、Criticの出力Layersと共有部分の逆伝播　NNの最初まで　終わり##
        
        return d_states              
        
    def train(self, episodes, steps_per_episode, gamma=0.99, metrics=1, standardize_G=True, softplus_to_advantage=False, 
              weight_decay_lmd=0, verbose_interval=100):
        
        start_time = datetime.now()
        
        #エピソード毎の記録list生成
        loss_episodes = [] #エピソード毎のlossのlist
        loss_actor_episodes = [] #エピソード毎のActorのlossのlist
        loss_critic_episodes = [] #エピソード毎のCriticのlossのlist
        step_count_episodes = [] #エピソード毎のステップ数のlist
        score_episodes = [] #エピソード毎のスコア（Σステップ毎の即時reward）のlist
        
        episode_count = 0
        step_count_total = 0
        best_step_count = 0 #今までのepisodeの中で最も多いstep回数
        best_score = -np.inf #今までのepisodeの中で最も大きいscore
        best_metrics_count = 0
                
        #エピソード反復        
        for ep in range(0, episodes):
            #1エピソード
            
            #このエピソード中の全ステップ記録のlist生成
            state_steps_episode = []
            action_steps_episode = []
            reward_steps_episode = []
            next_state_steps_episode = []
            
            step_count_ep = 0
            score_ep = 0
            save_temp_params = False
            
            #環境の開始時点にagentをセット
            curr_st = self._env.reset()  
            
            #ステップ反復
            for st in range(steps_per_episode):
                #1ステップ
                
                #ステップ実行
                action_predicted = self.predict_best_action(curr_st)
                next_st, reward_earned, done, info = self._env.step(action_predicted)
                
                #このエピソード中の全ステップ記録のlistに追加
                state_steps_episode.append(curr_st)
                action_steps_episode.append(action_predicted)
                reward_steps_episode.append(reward_earned)
                next_state_steps_episode.append(next_st)
                
                step_count_ep += 1
                step_count_total += 1
                
                #ステップ反復終了判定　steps_per_episodeに達したかenvからdone=Trueが返ってきたか
                #終了するならbreak
                if step_count_ep>=steps_per_episode or done==True:
                    break
                else:
                    curr_st = next_st                    
                    
            #ステップ反復　for　終わり
            
            #このエピソードのスコア算出
            #各ステップの即時報酬の単純合計
            score_ep = sum(reward_steps_episode)
            
            #このエピソードの”成績”が過去最高なら、パラメーターを退避
            
            if metrics==0 and (step_count_ep>=best_step_count):
                #step countで成績を計測
                best_step_count = step_count_ep
                best_metrics_count += 1
                save_temp_params = True
            elif metrics==1 and (score_ep>=best_score):
                #scoreで成績を計測
                best_score = score_ep
                best_metrics_count += 1
                save_temp_params = True
                
            if save_temp_params==True:
                #パラメーターを一時退避
                self._keep_temporarily_all_learnable_params()            
                        
            #割引報酬和「G」算出
            li_G = self._calc_G_of_step_in_an_episode(reward_steps_episode, gamma)
            
            #割引報酬和Gの標準化　引数standardize_Gでやるかやらないか指定される
            if standardize_G==True:
                Gs = standardize(li_G, with_mean=False).reshape(-1, 1) #平均を0にしない標準化
            else:
                Gs = np.array(li_G).reshape(-1, 1)
            
            #loss算出の順伝播            
            states = np.array(state_steps_episode)
            actions = np.array(action_steps_episode)
            
            loss_ep, loss_actor_ep, loss_critic_ep = self._forward_losses(
                state=states, action=actions, G=Gs, softplus_to_advantage=softplus_to_advantage, 
                weight_decay_lmd=weight_decay_lmd)
            
            #誤差逆伝播
            _ = self._backward_losses(d_L=1)
            
            #NN内の全learnable paramsの更新
            self._update_all_learnable_params(weight_decay_lmd)
            
            #このエピソードの記録をエピソード毎の記録listに追加
            step_count_episodes.append(step_count_ep) #ステップ数
            loss_episodes.append(loss_ep) #loss
            loss_actor_episodes.append(loss_actor_ep) #Actorのloss
            loss_critic_episodes.append(loss_critic_ep) #Criticのloss
            score_episodes.append(score_ep) #score
            
            if verbose_interval>0 and ( (ep+1)%verbose_interval==0 or ep==0 or (ep+1)==episodes ):

                time_string = datetime.now().strftime('%H:%M:%S')

                if metrics==0:
                    best_metrics_string = " best step count:" + str(best_step_count) + str(best_step_count) + "(" + str(best_metrics_count) + "回)"
                elif metrics==1:
                    best_metrics_string = " best score:" + str(best_score) + "(" + str(best_metrics_count) + "回)"
                else:
                    best_metrics_string=""
                
                if save_temp_params==True:
                    params_saved_string = " ベストなパラメーターを一時退避"
                else:
                    params_saved_string = ""

                print("Episode:" + str(ep) + " score:" + str(score_ep) + " step count:" + str(step_count_ep) + \
                      " loss:" + str(loss_ep) + " loss_actor:" + str(loss_actor_ep) + " loss_critic:" + str(loss_critic_ep) + \
                      best_metrics_string + params_saved_string + " time:" + time_string)               
                
            episode_count += 1
        
        #エピソード反復　for　終わり 
        
        #一時退避させてたパラメーターを戻す
        self._adopt_all_learnable_params_kept_temporarily()
        if verbose_interval>0:
            print("一時退避したベストなパラメーターを正式採用")
            
        end_time = datetime.now()        
        
        processing_time_total = end_time - start_time #総処理時間　datetime.timedelta
        processing_time_total_string = timedelta_HMS_string(processing_time_total) #総処理時間の文字列表現
        
        #resultを生成し、エピソード毎の記録listや引数やらを追加
        result = {}
        result["name"] = self._name
        result["episode_count"] = episode_count
        result["loss_episodes"] = loss_episodes
        result["loss_actor_episodes"] = loss_actor_episodes
        result["loss_critic_episodes"] = loss_critic_episodes
        result["step_count_episodes"] = step_count_episodes
        result["score_episodes"] = score_episodes
        result["step_count_total"] = step_count_total
        result["processing_time_total_string"] = processing_time_total_string
        result["processing_time_total"] = processing_time_total
        #以下引数
        result["episodes"] = episodes
        result["steps_per_episode"] = steps_per_episode
        result["gamma"] = gamma
        result["metrics"] = metrics
        result["standardize_G"] = standardize_G
        result["softplus_to_advantage"] = softplus_to_advantage
        result["weight_decay_lmd"] = weight_decay_lmd
        #以下メンバー変数
        result["wc_critic_loss"] = self._loss.wc
        
        return result
    
    def _calc_G_of_step_in_an_episode(self, li_reward_in_an_episode, gamma):
        #エピソード1個分の記録中の各ステップの割引報酬和Gを計算する。
        #li_reward_in_an_episode：エピソード1個分の全ステップのrewardのlist。
        #戻り値は、このエピソード1個の中の各ステップのsum_of_discounted_rewardsのlist。listのlenはステップ数。
        #※もしExperience Replay方式にする場合、ここは、エピソード1個分の全ステップ記録にGを付加し、そのエピソード1個分の記録を
        #経験バッファに追加、となる。
        
        li_G = []
        for t, r in enumerate(li_reward_in_an_episode):            
            li_discounted_rewards_future = \
            [r_following * (gamma**t_diff) for t_diff, r_following in enumerate(li_reward_in_an_episode[t:])]
            sum_of_discounted_rewards = sum(li_discounted_rewards_future)
            li_G.append(sum_of_discounted_rewards)
        
        return li_G
    
    def _update_all_learnable_params(self, weight_decay_lmd):
        #共有部分：trainableな全Layersのtrainableなパラメーターを一括更新する。
        #ActorとCriticの出力部分：trainableなパラメーターを一括更新する。
        
        #共有部分
        for layer in self._layers_shared.values():
            if layer.trainable == True:
                layer.update_learnable_params(weight_decay_lmd)
                
        #ActorとCriticの出力layers
        self._out_actor_mu.update_learnable_params(weight_decay_lmd)
        self._out_actor_log_var.update_learnable_params(weight_decay_lmd)
        self._out_critic.update_learnable_params(weight_decay_lmd)
        
    def save_params_in_file(self, file_dir, file_name=""):
        #このモデルインスタンスのパラメーターをファイル保存する。
        
        if file_name=="":
            file_name = self._name + ".pickle"
        
        file_path = file_dir + file_name
        
        #Dictionaryにして保存。keyはlayer.name。
        #all_params_dic(Dictionary)
        # --learnable layer1の保存したいパラメーターのtuple(weightsのndarray, biasesのndarray)
        # --learnable layer2の保存したいパラメーターのtuple(weightsのndarray, biasesのndarray)
        #　・
        #　・
        # --Actor outout layersの保存したいパラメーターのtuple(weightsのndarray, biasesのndarray)
        # --Critic outout layerの保存したいパラメーターのtuple(weightsのndarray, biasesのndarray)
        # --Loss_Compoundedの保存したいパラメーターのtuple(wc_critic_loss)

        all_params_dic = {}
            
        #trainableな全layerのlearnableなパラメーター分
        for layer in self._layers_shared.values():
            if layer.trainable == True:
                params_tpl = layer.copy_params()
                all_params_dic[layer.name] = params_tpl
            
        #Actor outout layersのパラメーター分
        all_params_dic[self._out_actor_mu.name] = self._out_actor_mu.copy_params()
        all_params_dic[self._out_actor_log_var.name] = self._out_actor_log_var.copy_params()
        
        #Critic outout layerのパラメーター分
        all_params_dic[self._out_critic.name] = self._out_critic.copy_params()
        
        #Loss_Compoundedのパラメーター分
        all_params_dic[self._loss.name] = self._loss.copy_params()
        
        save_pickle_file(all_params_dic, file_path)
        
        return file_name
        
    def overwrite_params_from_file(self, file_path):
        #このモデルインスタンスの全パラメーターを、ファイル保存されている別の物に差し替える。
        
        param_layer_tpls_dic = read_pickle_file(file_path)
        
        for layer_name in param_layer_tpls_dic.keys():

            #上書きするパラメーターをLayer毎に取り出す
            layer_params_tpl = param_layer_tpls_dic[layer_name] 

            #コピー先Layer毎に上書きする。
            #名前が同じLayerがコピー先Layer。
            
            if layer_name in self._layers_shared.keys():
                
                to_layer = self._layers_shared[layer_name]                
                if to_layer.trainable==True:
                    #ltrainableなLayer　上書き
                    to_layer.overwrite_params(layer_params_tpl)
                    
            elif layer_name==self._out_actor_mu.name:
                self._out_actor_mu.overwrite_params(layer_params_tpl)
            elif layer_name==self._out_actor_log_var.name:
                self._out_actor_log_var.overwrite_params(layer_params_tpl)
            elif layer_name==self._loss.name:
                self._loss.overwrite_params(layer_params_tpl)
                
    def _keep_temporarily_all_learnable_params(self):

        #配下の各trainableなLayerに対し、現時点でのlearnableパラメーターの一時退避を指示
        
        #共有部分
        for layer in self._layers_shared.values():
            if layer.trainable == True:
                layer.keep_temporarily_learnable_params()
                
        #Actorの出力layers
        self._out_actor_mu.keep_temporarily_learnable_params()
        self._out_actor_log_var.keep_temporarily_learnable_params()
        
        #Criticの出力layer
        self._out_critic.keep_temporarily_learnable_params()        

    def _adopt_all_learnable_params_kept_temporarily(self):

        #配下の各trainableなLayerに対し、一時退避していたlearnableパラメーターの正式採用を指示
        
        #共有部分
        for layer in self._layers_shared.values():
            if layer.trainable == True:
                layer.adopt_learnable_params_kept_temporarily()
                
        #Actorの出力layers
        self._out_actor_mu.adopt_learnable_params_kept_temporarily()
        self._out_actor_log_var.adopt_learnable_params_kept_temporarily()
        
        #Criticの出力layer
        self._out_critic.adopt_learnable_params_kept_temporarily() 
    
    def _sum_all_weights_square(self):
        #weightを持つtrainableな全Layerと出力layersのweightの2乗の総和を返す。
        #荷重減衰（weight decay）のため。

        sum_of_weights_square = 0
        for layer in self._layers_shared.values():
            if layer.trainable == True and isinstance(layer, Affine):
                sum_of_weights_square += layer.sum_weights_square()
                
        sum_of_weights_square += self._out_actor_mu.sum_weights_square()
        sum_of_weights_square += self._out_actor_log_var.sum_weights_square()
        sum_of_weights_square += self._out_critic.sum_weights_square()        

        return sum_of_weights_square
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def env(self):
        return self._env
    
    @property
    def state_dim(self):
        return self._state_dim
    
    @property
    def action_dim(self):
        return self._action_dim
    
    @property
    def wc_critic_loss(self):
        #wc_critic_lossを返す。
        return self._loss.wc
    
    @wc_critic_loss.setter
    def wc_critic_loss(self, wc_critic_loss):
         self._loss.wc = wc_critic_loss