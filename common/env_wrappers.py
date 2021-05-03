# -*- coding: utf-8 -*-
import gym


class EnvWrapper_plain(gym.Wrapper):
    """
    報酬設計の変更はせず、ラップされるenvと同様の挙動をする
    """
    def __init__(self, env):
        
        super(EnvWrapper_plain, self).__init__(env)
        
        self.env = env
        self.steps = 0        

    def reset(self):
        
        obs = self.env.reset()
        self.steps = 0
        
        return obs
    
    def step(self, action):
        
        obs, rew_, done, info, _ = self.step_ex(action)
        
        return obs, rew_, done, info

    def step_ex(self, action):
                
        self.steps += 1
        obs, rew, done, info = self.env.step(action)
            
        return obs, rew, done, info, rew
    

class EnvWrapper_01(gym.Wrapper):
    """
    エピソード終了時の報酬
     エピソード上限ステップ数まで到達したら報酬X, 到達しなければ報酬Y
     ステップごとの報酬はオリジナルのenvの報酬
    を返す
    """
    def __init__(self, env, max_steps_episode, reward_for_successful_episode=1, reward_for_failed_episode=-1):
        
        super(EnvWrapper_01, self).__init__(env)
        
        self.env = env        
        self._max_steps_episode = max_steps_episode
        self._reward_for_successful_episode = reward_for_successful_episode
        self._reward_for_failed_episode = reward_for_failed_episode
        
        self.steps = 0        

    def reset(self):
        
        obs = self.env.reset()
        self.steps = 0
        
        return obs
    
    def step(self, action):
        
        obs, rew_, done, info, _ = self.step_ex(action)
        
        return obs, rew_, done, info

    def step_ex(self, action):
        
        self.steps += 1
        obs, rew, done, info = self.env.step(action)
        
        rew_ = rew
        
        if done:
            if self.steps == self._max_steps_episode:
                if self._reward_for_successful_episode is not None:
                    rew_ = self._reward_for_successful_episode
            else:
                if self._reward_for_failed_episode is not None:
                    rew_ = self._reward_for_failed_episode
        else:
            rew_ = rew
            
        return obs, rew_, done, info, rew
    
    
class EnvWrapper_02(gym.Wrapper):
    """
    エピソード終了時の報酬 = x * 累積ステップ数
    ステップごとの報酬 = オリジナルのenvの報酬
    を返す
    """
    def __init__(self, env, reward_for_steps_episode=0.005):
        
        super(EnvWrapper_02, self).__init__(env)
        
        self.env = env        
        self._reward_for_steps_episode = reward_for_steps_episode
        
        self.steps = 0        

    def reset(self):
        
        obs = self.env.reset()
        self.steps = 0
        
        return obs
    
    def step(self, action):
        
        obs, rew_, done, info, _ = self.step_ex(action)
        
        return obs, rew_, done, info

    def step_ex(self, action):
        
        self.steps += 1
        obs, rew, done, info = self.env.step(action)
        
        if done:
            rew_ = self._reward_for_steps_episode * self.steps
        else:
            rew_ = rew
            
        return obs, rew_, done, info, rew
    
class EnvWrapper_03(gym.Wrapper):
    """
    エピソード終了時の報酬 = x * 累積ステップ数 + オリジナルのenvの報酬
    ステップごとの報酬 = オリジナルのenvの報酬
    を返す
    """
    def __init__(self, env, reward_for_steps_episode_added=0.005):
        
        super(EnvWrapper_03, self).__init__(env)
        
        self.env = env        
        self._reward_for_steps_episode_added = reward_for_steps_episode_added
        
        self.steps = 0        

    def reset(self):
        
        obs = self.env.reset()
        self.steps = 0
        
        return obs
    
    def step(self, action):
        
        obs, rew_, done, info, _ = self.step_ex(action)
        
        return obs, rew_, done, info

    def step_ex(self, action):
        
        self.steps += 1
        obs, rew, done, info = self.env.step(action)
        
        if done:
            rew_ = self._reward_for_steps_episode_added * self.steps + rew
        else:
            rew_ = rew
            
        return obs, rew_, done, info, rew
    
class EnvWrapper_04(gym.Wrapper):
    """
    単純に報酬をx倍して返す
    """
    def __init__(self, env, reward_scale=0.1):
        
        super(EnvWrapper_04, self).__init__(env)
        
        self.env = env        
        self._reward_scale = reward_scale
        
        self.steps = 0        

    def reset(self):
        
        obs = self.env.reset()
        self.steps = 0
        
        return obs
    
    def step(self, action):
        
        obs, rew_, done, info, _ = self.step_ex(action)
        
        return obs, rew_, done, info

    def step_ex(self, action):
        
        self.steps += 1
        obs, rew, done, info = self.env.step(action)
        
        rew_ = rew * self._reward_scale
            
        return obs, rew_, done, info, rew

class EnvWrapper_05(gym.Wrapper):
    """
    エピソード終了時の報酬
     エピソード成功時は報酬X, 失敗時は報酬Y
     成功/失敗の判定：エピソード終了時の報酬がoriginal_reward_for_failed_episodeなら失敗
     ステップごとの報酬はオリジナルのenvの報酬
    を返す
    """
    def __init__(self, env, original_reward_for_failed_episode, reward_for_successful_episode=1, reward_for_failed_episode=-1):
        
        super(EnvWrapper_05, self).__init__(env)
        
        self.env = env        
        self._original_reward_for_failed_episode = original_reward_for_failed_episode
        self._reward_for_successful_episode = reward_for_successful_episode
        self._reward_for_failed_episode = reward_for_failed_episode
        
        self.steps = 0        

    def reset(self):
        
        obs = self.env.reset()
        self.steps = 0
        
        return obs
    
    def step(self, action):
        
        obs, rew_, done, info, _ = self.step_ex(action)
        
        return obs, rew_, done, info

    def step_ex(self, action):
        
        self.steps += 1
        obs, rew, done, info = self.env.step(action)
        
        rew_ = rew
        
        if done:
            if rew == self._original_reward_for_failed_episode:
                if self._reward_for_failed_episode is not None:
                    rew_ = self._reward_for_failed_episode                
            else:
                if self._reward_for_successful_episode is not None:
                    rew_ = self._reward_for_successful_episode
        else:
            rew_ = rew
            
        return obs, rew_, done, info, rew