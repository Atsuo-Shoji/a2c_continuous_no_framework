# -*- coding: utf-8 -*-
import numpy as np
from collections import deque, OrderedDict
from datetime import datetime
from common.funcs import *
from common.layers import *
from common.optimizers import *

class Planner_separate():
    
    def __init__(self, name, env, state_dim, action_dim):
        
        #name：このPlannerインスタンスの名前
        #env：このPlannerインスタンスが対象とする環境
        #state_dim：このPlannerインスタンスが認識する状態の要素数
        #action_dim：このPlannerインスタンスが認識する行動の要素数
        
        self._name = name
        self._env = env
        self._state_dim = state_dim
        self._action_dim = action_dim
        
        ###Layersの定義###
        
        output_dim_afn1 = self._state_dim*10
        output_dim_afn3 = self._action_dim*10
        output_dim_afn2 = np.ceil( np.sqrt(output_dim_afn1*output_dim_afn3) ).astype(np.int)

        ##ActorのLayersの定義##
        
        self._layers_actor = OrderedDict()
        
        #①Affine 「afn1_actor」　instance of Affine
        opt_afn1_actor = Adam(lr=0.001, rho1=0.9, rho2=0.999)        
        afn1_actor = Affine(name="afn1_actor", input_shape=(state_dim,), output_shape=(output_dim_afn1,), optimizer=opt_afn1_actor, 
                        init_weight_option="xavier")
        self._layers_actor[afn1_actor.name] = afn1_actor
        prev_layer = afn1_actor

        #②Tanh　「tanh_afn1_actor」　instance of Tanh
        tanh_afn1_actor = Tanh(name="tanh_afn1_actor", input_shape=prev_layer.output_shape)
        self._layers_actor[tanh_afn1_actor.name] = tanh_afn1_actor
        prev_layer = tanh_afn1_actor

        #③Affine 「afn2_actor」　instance of Affine
        opt_afn2_actor = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        afn2_actor = Affine(name="afn2_actor", input_shape=prev_layer.output_shape, output_shape=(output_dim_afn2,), optimizer=opt_afn2_actor, 
                        init_weight_option="xavier")
        self._layers_actor[afn2_actor.name] = afn2_actor
        prev_layer = afn2_actor

        #④Tanh　「tanh_afn2_actor」　instance of Tanh
        tanh_afn2_actor = Tanh(name="tanh_afn2_actor", input_shape=prev_layer.output_shape)
        self._layers_actor[tanh_afn2_actor.name] = tanh_afn2_actor
        prev_layer = tanh_afn2_actor
        
        #⑤Affine 「afn3_actor」　instance of Affine
        opt_afn3_actor = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        afn3_actor = Affine(name="afn3_actor", input_shape=prev_layer.output_shape, output_shape=(output_dim_afn3,), optimizer=opt_afn3_actor, 
                        init_weight_option="xavier")
        self._layers_actor[afn3_actor.name] = afn3_actor
        prev_layer = afn3_actor

        #⑥Tanh　「tanh_afn3_actor」　instance of Tanh
        tanh_afn3_actor = Tanh(name="tanh_afn3_actor", input_shape=prev_layer.output_shape)
        self._layers_actor[tanh_afn3_actor.name] = tanh_afn3_actor        
        prev_layer = tanh_afn3_actor
        
        #Actor　平均μの出力のlayers　中身はAffine1個とTanh1個
        opt_afn_actor_mu = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        self._out_actor_mu = Actor_Output(name="out_actor_mu", input_shape=prev_layer.output_shape, 
                                          output_shape=(action_dim,), optimizer=opt_afn_actor_mu)
        #self._layers_actorに含めないので注意
        
        #Actor　分散varのlog(var)の出力のlayers　中身はAffine1個とTanh1個
        opt_afn_actor_log_var = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        self._out_actor_log_var = Actor_Output(name="out_actor_log_var", input_shape=prev_layer.output_shape, 
                                               output_shape=(action_dim,), optimizer=opt_afn_actor_log_var)
        #self._layers_actorに含めないので注意
        
        ##ActorのLayersの定義　終わり##
        
        ##CriticのLayersの定義##
        
        self._layers_critic = OrderedDict()
        
        #①Affine 「afn1_critic」　instance of Affine
        opt_afn1_critic = Adam(lr=0.001, rho1=0.9, rho2=0.999)        
        afn1_critic = Affine(name="afn1_critic", input_shape=(state_dim,), output_shape=(output_dim_afn1,), optimizer=opt_afn1_critic, 
                        init_weight_option="xavier")
        self._layers_critic[afn1_critic.name] = afn1_critic
        prev_layer = afn1_critic

        #②Tanh　「tanh_afn1_critic」　instance of Tanh
        tanh_afn1_critic = Tanh(name="tanh_afn1_critic", input_shape=prev_layer.output_shape)
        self._layers_critic[tanh_afn1_critic.name] = tanh_afn1_critic
        prev_layer = tanh_afn1_critic

        #③Affine 「afn2_critic」　instance of Affine
        opt_afn2_critic = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        afn2_critic = Affine(name="afn2_critic", input_shape=prev_layer.output_shape, output_shape=(output_dim_afn2,), optimizer=opt_afn2_critic, 
                        init_weight_option="xavier")
        self._layers_critic[afn2_critic.name] = afn2_critic
        prev_layer = afn2_critic

        #④Tanh　「tanh_afn2_critic」　instance of Tanh
        tanh_afn2_critic = Tanh(name="tanh_afn2_critic", input_shape=prev_layer.output_shape)
        self._layers_critic[tanh_afn2_critic.name] = tanh_afn2_critic
        prev_layer = tanh_afn2_critic
        
        #⑤Affine 「afn3_critic」　instance of Affine
        opt_afn3_critic = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        afn3_critic = Affine(name="afn3_critic", input_shape=prev_layer.output_shape, output_shape=(output_dim_afn3,), optimizer=opt_afn3_critic, 
                        init_weight_option="xavier")
        self._layers_critic[afn3_critic.name] = afn3_critic
        prev_layer = afn3_critic

        #⑥Tanh　「tanh_afn3_critic」　instance of Tanh
        tanh_afn3_critic = Tanh(name="tanh_afn3_critic", input_shape=prev_layer.output_shape)
        self._layers_critic[tanh_afn3_critic.name] = tanh_afn3_critic        
        prev_layer = tanh_afn3_critic
        
        #⑦Affine 「afn4_critic」　instance of Affine　出力layer　活性化関数は無し（linear）
        opt_afn4_critic = Adam(lr=0.001, rho1=0.9, rho2=0.999)
        afn4_critic = Affine(name="afn4_critic", input_shape=prev_layer.output_shape, output_shape=(1,), optimizer=opt_afn4_critic, 
                        init_weight_option="xavier")
        self._layers_critic[afn4_critic.name] = afn4_critic
        
        ##CriticのLayersの定義　終わり##
        
        #ActorとCritic　Lossの定義##
        
        #ActorのLoss
        self._loss_actor = Actor_Loss("loss_actor")
        
        #criticのLoss
        self._loss_critic = Critic_Loss("loss_critic")
        
        ##ActorとCritic　Lossの定義　終わり##
                
    def predict_best_action(self, a_state):
        #最適な行動を推測
        
        if a_state.ndim==1:
            a_state = a_state.reshape(1, self._state_dim) 
        elif a_state.shape[0]!=1:
            raise ValueException("a_stateは1件だけにしてください。1件のstateについてbest actionを推測します。")            
        
        #Actorから正規分布の平均μと分散varのlog_varを得る。
        #どちらも(1, action_dim) 
        mu, log_var = self._forward_output_actor(a_state, train_flg=False)
        
        #分散var
        #(1, action_dim) 
        var = np.exp(log_var)
        
        #muとvarからバッチ数のaxis=0を削除し、(action_dim,)にする
        mu = mu[0]
        var = var[0]
        
        #平均μと分散varの正規分布からサンプリング
        best_action = np.random.normal(loc=mu, scale=np.sqrt(var))
        
        return best_action
    
    def _forward_output_actor(self, state, train_flg):
        #Actorの出力
        #分岐以外の、勾配に影響を与える演算はしないこと
        
        if state.ndim==1:
            state = state.reshape(1, self._state_dim)  
            
        ##Actor中間layersの出力##        
        #出力値を「m」とする
        #(N, M) 「M」は最後の中間layerのAffineのニューロン数
        x = state
        for layer in self._layers_actor.values():
            x = layer.forward(x, train_flg)            
        m = x
        ##Actor中間layersの出力　終わり##
        
        ##Actorの出力##
        
        #2つの出力layersにActor中間layersの出力を渡す。
        #Actor中間layersの出力が2分岐する。
        
        #Actorのμ出力layersからの出力 (N, action_dim)
        mu = self._out_actor_mu.forward(m, train_flg)
        
        #Actorのlog(var)出力layersからの出力 (N, action_dim)
        log_var = self._out_actor_log_var.forward(m, train_flg)        
        
        ##Actorの出力　終わり##
        
        return mu, log_var
    
    def _backward_output_actor(self, d_mu, d_log_var):
        #Actorの出力から最初までの誤差逆伝播
        
        ##Actorの出力の逆伝播##
        
        #Actorのμ出力部分の逆伝播
        #(N, M) 「M」は共有最終layerのAffineのニューロン数
        d_m_mu = self._out_actor_mu.backward(d_mu)
        
        #Actorのlog(var)出力部分の逆伝播
        #(N, M) 「M」はActor中間layersの最終layerのAffineのニューロン数
        d_m_log_var = self._out_actor_log_var.backward(d_log_var)
        
        #forwardでActor中間layersの最終layerの出力mを
        #Actorのμ出力、Actorのlog(var)出力
        #に2分岐したので、これらから逆伝播してきた勾配を合算する。
        d_m = d_m_mu + d_m_log_var
        
        ##Actorの出力の逆伝播　終わり##
        
        ##Actor中間layers（最初まで）の逆伝播##
        #(N, state_dim)
        layers_shared = list(self._layers_actor.values())
        layers_shared.reverse()
        d_out = d_m
        for layer in layers_shared:
            d_out = layer.backward(d_out)        
        d_state = d_out
        ##Actor中間layers（最初まで）の逆伝播　終わり##
        
        return d_state
    
    def _forward_output_critic(self, state, train_flg):
        #Criticの出力
        #分岐以外の、勾配に影響を与える演算はしないこと
        
        if state.ndim==1:
            state = state.reshape(1, self._state_dim)  
            
        x = state
        for layer in self._layers_critic.values():
            x = layer.forward(x, train_flg)            
        V = x
                
        return V
    
    def _backward_output_critic(self, d_V):
        #Criticの出力から最初までの誤差逆伝播
        
        layers_shared = list(self._layers_critic.values())
        layers_shared.reverse()
        d_out = d_V
        for layer in layers_shared:
            d_out = layer.backward(d_out)        
        d_state = d_out
        
        return d_state
    
    def _forward_loss_actor(self, state, action, G, V, softplus_to_advantage, weight_decay_lmd):
        #Actorのloss
        #action, Gのもとになったstateは、引数stateと同じ並びであること
        #分岐以外の、勾配に影響を与える演算はしないこと
        #V：Criticの出力　(N, 1)
        
        ##Actorの出力##        
        #Actorの出力μ、Actorの出力log(var)
        #(N, action_dim), (N, action_dim), 
        mu, log_var = self._forward_output_actor(state, train_flg=True)        
        ##Actorの出力　終わり##
        
        ##Actorのloss##        
        La = self._loss_actor.forward(mu, log_var, action, V, G, softplus_to_advantage)            
        ##Actorのloss　終わり##
        
        #荷重減衰
        if weight_decay_lmd > 0:
            sum_all_weights_square = self._sum_all_weights_square(role=0)
            La = La + 0.5*weight_decay_lmd*sum_all_weights_square
        
        return La
    
    def _backward_loss_actor(self, d_La=1):
        #Actorのlossから最初までの一気通貫の誤差逆伝播
        
        ##Actorのloss　逆伝播##        
        #どちらも(N, action_dim)
        d_mu, d_log_var = self._loss_actor.backward(d_La)        
        ##Actorのloss　逆伝播　終わり##
        
        ##Actorの出力と中間Layersの逆伝播　最初まで##
        #(N, state_dim)
        d_states = self._backward_output_actor(d_mu, d_log_var)
        ##Actorの出力と中間Layersの逆伝播　最初まで　終わり##
        
        return d_states
    
    def _forward_loss_critic(self, state, G, weight_decay_lmd):
        #Criticのloss
        #Gのもとになったstateは、引数stateと同じ並びであること
        #分岐以外の、勾配に影響を与える演算はしないこと
        
        ##Criticの出力##        
        #Criticの出力V
        #(N, 1), 
        V = self._forward_output_critic(state, train_flg=True)        
        ##Criticの出力　終わり##
        
        ##Criticのloss##        
        Lc = self._loss_critic.forward(V, G)  #Gは教師信号として使用される
        ##Criticのloss　終わり##
        
        #荷重減衰
        if weight_decay_lmd > 0:
            sum_all_weights_square = self._sum_all_weights_square(role=1)
            Lc = Lc + 0.5*weight_decay_lmd*sum_all_weights_square
        
        #戻り値にVがあるのは、呼び出し元で使うから
        return Lc, V
    
    def _backward_loss_critic(self, d_Lc=1):
        #Criticのlossから最初までの一気通貫の誤差逆伝播
        
        ##Criticのloss　逆伝播##        
        #(N, 1)
        d_V = self._loss_critic.backward(d_Lc)        
        ##Criticのloss　逆伝播#　終わり##
        
        ##Criticの出力と中間Layersの逆伝播　最初まで##
        #(N, state_dim)
        d_states = self._backward_output_critic(d_V)
        ##Criticの出力と中間Layersの逆伝播　最初まで　終わり##
        
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
            
            #ActorとCriticのloss算出の順伝播　引数の準備       
            states = np.array(state_steps_episode)
            actions = np.array(action_steps_episode)
            
            #Critic　loss算出の順伝播　Actorで使用するVも取得
            loss_critic_ep, Vs = self._forward_loss_critic(state=states, G=Gs, weight_decay_lmd=weight_decay_lmd)
            #Critic　逆伝播
            _ = self._backward_loss_critic(d_Lc=1)
            #Critic　パラメーター更新
            self._update_all_learnable_params(role=1, weight_decay_lmd=weight_decay_lmd)
            
            #Actor　loss算出の順伝播
            loss_actor_ep = self._forward_loss_actor(state=states, action=actions, G=Gs, V=Vs, 
                                                     softplus_to_advantage=softplus_to_advantage, 
                                                     weight_decay_lmd=weight_decay_lmd)
            #Actor　逆伝播
            _ = self._backward_loss_actor(d_La=1)
            #Actor　パラメーター更新
            self._update_all_learnable_params(role=0, weight_decay_lmd=weight_decay_lmd)
        
            #このエピソードの記録をエピソード毎の記録listに追加
            step_count_episodes.append(step_count_ep) #ステップ数
            loss_actor_episodes.append(loss_actor_ep) #Actorのloss
            loss_critic_episodes.append(loss_critic_ep) #Criticのloss
            score_episodes.append(score_ep) #score
            
            if verbose_interval>0 and ( (ep+1)%verbose_interval==0 or ep==0 or (ep+1)==episodes ):

                time_string = datetime.now().strftime('%H:%M:%S')

                if metrics==0:
                    best_metrics_string = " best step count:" + str(best_step_count) + "(" + str(best_metrics_count) + "回)"
                elif metrics==1:
                    best_metrics_string = " best score:" + str(best_score) + "(" + str(best_metrics_count) + "回)"
                else:
                    best_metrics_string=""
                
                if save_temp_params==True:
                    params_saved_string = " ベストなパラメーターを一時退避"
                else:
                    params_saved_string = ""

                print("Episode:" + str(ep) + " score:" + str(score_ep) + " step count:" + str(step_count_ep) + \
                      " loss_actor:" + str(loss_actor_ep) + " loss_critic:" + str(loss_critic_ep) + \
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
        result["softplus_to_advantage"] = softplus_to_advantage
        result["weight_decay_lmd"] = weight_decay_lmd
        
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
    
    def _update_all_learnable_params(self, role, weight_decay_lmd):
        #role：Actor（0）かCritic（1）か
        #指定されたroleの全Layersのtrainableなパラメーターを一括更新する。
        
        if role==0:
            #Actor
            for layer in self._layers_actor.values():
                if layer.trainable == True:
                    layer.update_learnable_params(weight_decay_lmd)
            self._out_actor_mu.update_learnable_params(weight_decay_lmd)
            self._out_actor_log_var.update_learnable_params(weight_decay_lmd)
        elif role==1:
            #Critic
            for layer in self._layers_critic.values():
                if layer.trainable == True:
                    layer.update_learnable_params(weight_decay_lmd)
                    
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
        
        all_params_dic = {}
            
        #Actorのパラメーター分
        for layer in self._layers_actor.values():
            if layer.trainable == True:
                params_tpl = layer.copy_params()
                all_params_dic[layer.name] = params_tpl
        all_params_dic[self._out_actor_mu.name] = self._out_actor_mu.copy_params()
        all_params_dic[self._out_actor_log_var.name] = self._out_actor_log_var.copy_params()
        
        #Criticのパラメーター分
        for layer in self._layers_critic.values():
            if layer.trainable == True:
                params_tpl = layer.copy_params()
                all_params_dic[layer.name] = params_tpl
        
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
            
            if layer_name in self._layers_actor.keys():
                #Actor　中間layers
                to_layer = self._layers_actor[layer_name]                
                if to_layer.trainable==True:
                    #ltrainableなLayer　上書き
                    to_layer.overwrite_params(layer_params_tpl)                               
            elif layer_name==self._out_actor_mu.name:
                #Actor　μ出力layers
                self._out_actor_mu.overwrite_params(layer_params_tpl)
            elif layer_name==self._out_actor_log_var.name:
                #Actor　log(var)出力layers
                self._out_actor_log_var.overwrite_params(layer_params_tpl)
            elif layer_name in self._layers_critic.keys(): 
                #Critic　全layers
                to_layer = self._layers_critic[layer_name]                
                if to_layer.trainable==True:
                    #ltrainableなLayer　上書き
                    to_layer.overwrite_params(layer_params_tpl)             
                
    def _keep_temporarily_all_learnable_params(self):

        #配下の各trainableなLayerに対し、現時点でのlearnableパラメーターの一時退避を指示
        
        #Actor
        for layer in self._layers_actor.values():
            if layer.trainable == True:
                layer.keep_temporarily_learnable_params()
        self._out_actor_mu.keep_temporarily_learnable_params()
        self._out_actor_log_var.keep_temporarily_learnable_params()
        
        #Critic
        for layer in self._layers_critic.values():
            if layer.trainable == True:
                layer.keep_temporarily_learnable_params()       

    def _adopt_all_learnable_params_kept_temporarily(self):

        #配下の各trainableなLayerに対し、一時退避していたlearnableパラメーターの正式採用を指示
        
        #Actor
        for layer in self._layers_actor.values():
            if layer.trainable == True:
                layer.adopt_learnable_params_kept_temporarily()
        self._out_actor_mu.adopt_learnable_params_kept_temporarily()
        self._out_actor_log_var.adopt_learnable_params_kept_temporarily()
        
        #Critic
        for layer in self._layers_critic.values():
            if layer.trainable == True:
                layer.adopt_learnable_params_kept_temporarily()        
        
    def _sum_all_weights_square(self, role):
        #role：Actor（0）かCritic（1）か
        #weightを持つtrainableな全Layerと出力layersのweightの2乗の総和を返す。
        #荷重減衰（weight decay）のため。
        
        if role==0:
            #Actor
            sum_of_weights_square = 0
            for layer in self._layers_actor.values():
                if layer.trainable == True and isinstance(layer, Affine):
                    sum_of_weights_square += layer.sum_weights_square()
            sum_of_weights_square += self._out_actor_mu.sum_weights_square()
            sum_of_weights_square += self._out_actor_log_var.sum_weights_square()            
        elif role==1:
            #Critic
            sum_of_weights_square = 0
            for layer in self._layers_critic.values():
                if layer.trainable == True and isinstance(layer, Affine):
                    sum_of_weights_square += layer.sum_weights_square()
        else:
            sum_of_weights_square = 0
                    
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