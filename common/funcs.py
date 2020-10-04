# -*- coding: utf-8 -*-
import numpy as np
import pickle


###Utility Functions###

def calculate_init_std_weight(n1, n2, option, default_init_std=0.1):

    #weightの初期値決定に使用する標準偏差の算出
    
    if option == "xavier":
        std = np.sqrt(2 / (n1 + n2))
    elif option == "He":
        std = np.sqrt(2 / n1)
    else:
        std = default_init_std
    return std

def read_pickle_file(file_path):
    #指定されたパスのpickleファイルを読み込む。
    
    with open(file_path, "rb") as fo:
        obj = pickle.load(fo)
        
    return obj
        
def save_pickle_file(obj, file_path):
    #指定されたオブジェクトを指定されたパスのpickleファイルとして書き込む。
    
    with open(file_path, 'wb') as fo:
        pickle.dump(obj , fo) 
        
def timedelta_HMS_string(td):
    
    hours = td.seconds//3600
    remainder_secs = td.seconds%3600
    minutes = remainder_secs//60
    seconds = remainder_secs%60
    
    HMS_string = str(hours) + " hours " + str(minutes) + " minutes " + str(seconds) + " seconds"
    
    return HMS_string

###Utility Functions　終わり###

