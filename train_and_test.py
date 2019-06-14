# -*- coding: utf-8 -*-
"""
Created on Fri May  4 19:17:15 2018

@author: ZhengZhiyong

@Target: 训练+预测

@Parameter:
    dataFileTrain = r'G:\pythonProject\discharge_db\B0036.dat' #训练集的文件路径
    dataFileTest = r'G:\pythonProject\discharge_db\B0048.dat'  #测试集的文件路径
    
@attention:    
    db['discharge'][cycle_num]['Voltage_measured']
    
"""

import random
from copy import deepcopy
import shelve #小型数据库
import matplotlib.pyplot as plt  #绘图
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import pandas
import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import preprocessing  #标准化处理
from random import sample #产生随机元素
#import xgboost as xgb  
from sklearn import neighbors #KNN回归
from sklearn import ensemble  #随机森林
from sklearn import neural_network



from zzy_model import extract_feature#导入自定义特征提取模块
from zzy_model import pca_down    
from zzy_model import rmse_calc
from zzy_model import mean_sudden_filter
from zzy_model import smooth
from zzy_model import precision_calc

#%特征数据预处理-------------------------------------------------------------------------------------------------------------
def feature_preprocessing1(temp_feature_list):
    #去除第一个点
    temp_feature_list=temp_feature_list[1:]
    #均值突变滤波
    temp_feature_list=mean_sudden_filter(temp_feature_list,3,2.2)   
    
    return temp_feature_list

def feature_preprocessing2(temp_feature_list):
    #去除第一个点
    temp_feature_list=temp_feature_list[1:]

    #均值突变滤波
    temp_feature_list=mean_sudden_filter(temp_feature_list,3,2.5)   
    
    return temp_feature_list
def extract_feature_from_file(db_file_list):
    """ 
    @Target:批量计算多个电池文件的特征
    
    @Parameter:
        db_file_list       #待处理的数据库文件列表  
        
        db_file_list=[r'G:\pythonProject\discharge_db\B0036.dat',
              r'G:\pythonProject\discharge_db\B0034.dat']
    @return
        battary_feature_dict_list[filenum]['maxV_list'] 一个包含字典的列表，字典对于的是具体特征
    @Eg:
        ALL_feature_from_file=extract_feature_from_file(db_file_list)
        plt.plot(ALL_feature_from_file[0]['VCE_list'])    
    """  
    #定义临时列表，缓存
    discharge_db_list=[]
    #定义整个电池文件的特征return缓存
    battary_feature_dict_list=[]   #最终输出的列表，每个元素是一个文件特征字典

    for num in range(len(db_file_list)):
        db = shelve.open(db_file_list[num], flag='c', protocol=None, writeback=False)
        discharge_db_list=deepcopy(db['discharge'])
        db.close() 
        
        #一个文件特征字典
        battary_feature_dict={}
        battary_feature_dict['maxV_list']=[]
        battary_feature_dict['TmaxV_list']=[]
        battary_feature_dict['minV_list']=[]
        battary_feature_dict['TminV_list']=[]
        battary_feature_dict['maxT_list']=[]
        battary_feature_dict['TmaxT_list']=[]
        battary_feature_dict['minT_list']=[]
        battary_feature_dict['TminT_list']=[]
        battary_feature_dict['Cap_list']=[]
        battary_feature_dict['VCE_list']=[]
        battary_feature_dict['TCE_list']=[]
        battary_feature_dict['VC_FI_list']=[]
        battary_feature_dict['TC_FI_list']=[]     
        battary_feature_dict['VC_CI_list']=[]
        battary_feature_dict['TC_CI_list']=[]
        battary_feature_dict['VC_SI_list']=[]  
        battary_feature_dict['TC_SI_list']=[]
        battary_feature_dict['VC_KI_list']=[]
        battary_feature_dict['TC_KI_list']=[]
        battary_feature_dict['Mean_V_list']=[]    
        battary_feature_dict['Mean_T_list']=[]
        #discharge_db_list[cycle_num]['Voltage_measured']     
        feature_dict={}  #每个放电周期的21个特征值缓存
        for i in range(len(discharge_db_list)):   #迭代一个电池的所有放电循环、
            extract_feature(discharge_db_list[i],feature_dict)    #函数内部迭代每个放电周期中的所有采样次数,每个采样周期都有21个特征值
            battary_feature_dict['maxV_list'].append(feature_dict['maxV'])
            battary_feature_dict['TmaxV_list'].append(feature_dict['TmaxV'])
            battary_feature_dict['minV_list'].append(feature_dict['minV'])
            battary_feature_dict['TminV_list'].append(feature_dict['TminV'])
            battary_feature_dict['maxT_list'].append(feature_dict['maxT'])
            battary_feature_dict['TmaxT_list'].append(feature_dict['TmaxT'])
            battary_feature_dict['minT_list'].append(feature_dict['minT'])
            battary_feature_dict['TminT_list'].append(feature_dict['TminT'])
            battary_feature_dict['Cap_list'].append(feature_dict['Cap'])     
            battary_feature_dict['VCE_list'].append(feature_dict['VCE'])
            battary_feature_dict['TCE_list'].append(feature_dict['TCE'])    
            battary_feature_dict['VC_FI_list'].append(feature_dict['VC_FI'])
            battary_feature_dict['TC_FI_list'].append(feature_dict['TC_FI'])     
            battary_feature_dict['VC_CI_list'].append(feature_dict['VC_CI'])
            battary_feature_dict['TC_CI_list'].append(feature_dict['TC_CI'])
            battary_feature_dict['VC_SI_list'].append(feature_dict['VC_SI'])      
            battary_feature_dict['TC_SI_list'].append(feature_dict['TC_SI'])
            battary_feature_dict['VC_KI_list'].append(feature_dict['VC_KI'])
            battary_feature_dict['TC_KI_list'].append(feature_dict['TC_KI'])
            battary_feature_dict['Mean_V_list'].append(feature_dict['Mean_V'])      
            battary_feature_dict['Mean_T_list'].append(feature_dict['Mean_T'])         
            
        #预处理
        battary_feature_dict['maxV_list']=feature_preprocessing1(battary_feature_dict['maxV_list'])  
        battary_feature_dict['TmaxV_list']=feature_preprocessing1(battary_feature_dict['TmaxV_list'])    
        battary_feature_dict['minV_list']=feature_preprocessing1(battary_feature_dict['minV_list'])
        battary_feature_dict['TminV_list']=feature_preprocessing1(battary_feature_dict['TminV_list'])
        battary_feature_dict['maxT_list']=feature_preprocessing1(battary_feature_dict['maxT_list'])
        battary_feature_dict['TmaxT_list']=feature_preprocessing1(battary_feature_dict['TmaxT_list'])
        battary_feature_dict['minT_list']=feature_preprocessing1(battary_feature_dict['minT_list'])
        battary_feature_dict['TminT_list']=feature_preprocessing1(battary_feature_dict['TminT_list'])
        battary_feature_dict['Cap_list']=feature_preprocessing1(battary_feature_dict['Cap_list'])
        battary_feature_dict['VCE_list']=feature_preprocessing1(battary_feature_dict['VCE_list'])
        battary_feature_dict['TCE_list']=feature_preprocessing1(battary_feature_dict['TCE_list'])
        battary_feature_dict['VC_FI_list']=feature_preprocessing2(battary_feature_dict['VC_FI_list'])
        battary_feature_dict['TC_FI_list']=feature_preprocessing1(battary_feature_dict['TC_FI_list'])
        battary_feature_dict['VC_CI_list']=feature_preprocessing1(battary_feature_dict['VC_CI_list'])
        battary_feature_dict['TC_CI_list']=feature_preprocessing1(battary_feature_dict['TC_CI_list'])
        battary_feature_dict['VC_SI_list']=feature_preprocessing1(battary_feature_dict['VC_SI_list'])
        battary_feature_dict['TC_SI_list']=feature_preprocessing1(battary_feature_dict['TC_SI_list'])
        battary_feature_dict['VC_KI_list']=feature_preprocessing1(battary_feature_dict['VC_KI_list'])
        battary_feature_dict['TC_KI_list']=feature_preprocessing1(battary_feature_dict['TC_KI_list'])
        battary_feature_dict['Mean_V_list']=feature_preprocessing1(battary_feature_dict['Mean_V_list'])
        battary_feature_dict['Mean_T_list']=feature_preprocessing1(battary_feature_dict['Mean_T_list'])    
        battary_feature_dict_list.append(battary_feature_dict)   #battary_feature_dict['maxV_list']=maxV_list[]   数据结构
   
    return battary_feature_dict_list

#%
#plt.plot(ALL_feature_from_file[0]['VC_FI_list'])    
#plt.plot(ALL_feature_from_file[1]['VC_FI_list'])   

#%得到以时间窗为单位的数据集和测试集
def extract_train_test_from_file_list_window(ALL_feature_from_file,window_size):
    """ 
    @Target:批量处理每个电池文件，对每个文件以时间窗为单位构特征样本，并从中随机提取70%作为训练集、30作为测试集
    
    @Parameter:
        ALL_feature_from_file       #extract_feature_from_file函数残生的列表（包含特征）  
        window_size                 #设置时间窗的大小
    @return
        一个列表，元素对应每个电池文件，元素[0]为train列表，元素[1]为test列表,每个列表代表一个特征样本，最后一个值表示RUL
    @Eg:
        extract_train_test_from_file_list_window=extract_train_test_from_file_list(ALL_feature_from_file,5)    
    """  
    all_train_test_list=[]   #定义一个lsit存放所有文件的输出
    for file_num in range(len(ALL_feature_from_file)):  #遍历所有个文件的特征字典 battary_feature_dict['maxV_list']
        
        train_test_list=[]   #每个电池文件产生的train和test 索引分别是0和1
        
        file_samples_windw={}   #定义一个字典存放该文件的所有基于窗口的特征样本【'maxV_windw_list'】【【第一个窗】【第二个窗】】
        file_samples_windw['maxV_windw_list']=[]
        file_samples_windw['TmaxV_windw_list']=[]
        file_samples_windw['minV_windw_list']=[]
        file_samples_windw['TminV_windw_list']=[]
        file_samples_windw['maxT_windw_list']=[]
        file_samples_windw['TmaxT_windw_list']=[]
        file_samples_windw['minT_windw_list']=[]
        file_samples_windw['TminT_windw_list']=[]
        file_samples_windw['Cap_windw_list']=[]
        file_samples_windw['VCE_windw_list']=[]
        file_samples_windw['TCE_windw_list']=[]
        file_samples_windw['VC_FI_windw_list']=[]
        file_samples_windw['TC_FI_windw_list']=[]     
        file_samples_windw['VC_CI_windw_list']=[]
        file_samples_windw['TC_CI_windw_list']=[]
        file_samples_windw['VC_SI_windw_list']=[]  
        file_samples_windw['TC_SI_windw_list']=[]
        file_samples_windw['VC_KI_windw_list']=[]
        file_samples_windw['TC_KI_windw_list']=[]
        file_samples_windw['Mean_V_windw_list']=[]    
        file_samples_windw['Mean_T_windw_list']=[]

        cycleNums=len(ALL_feature_from_file[file_num]['maxV_list'])
        #为文件的每种特征生成窗口
        #1.maxV
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['maxV_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['maxV_windw_list'].append(new_windw)        

        #2.TmaxV
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['TmaxV_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['TmaxV_windw_list'].append(new_windw)
        
        #3.minV
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['minV_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['minV_windw_list'].append(new_windw)        
        
        #4.TminV
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['TminV_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['TminV_windw_list'].append(new_windw)          
        
        #5.maxT
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['maxT_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['maxT_windw_list'].append(new_windw)            
        
        #6.TmaxT
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['TmaxT_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['TmaxT_windw_list'].append(new_windw)             
        
        #7.minT
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['minT_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['minT_windw_list'].append(new_windw)           
        
        #8.TminT
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['TminT_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['TminT_windw_list'].append(new_windw)           
        
        #9.Cap
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['Cap_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['Cap_windw_list'].append(new_windw)          
        
        #10.VCE
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['VCE_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['VCE_windw_list'].append(new_windw)           
        
        #11.TCE
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['TCE_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['TCE_windw_list'].append(new_windw)            
        
        #12.VC_FI
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['VC_FI_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['VC_FI_windw_list'].append(new_windw)                
        
        #13.TC_FI
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['TC_FI_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['TC_FI_windw_list'].append(new_windw)            
        
        #14.VC_CI
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['VC_CI_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['VC_CI_windw_list'].append(new_windw)             
        
        #15.TC_CI
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['TC_CI_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['TC_CI_windw_list'].append(new_windw)         
            
        #16.VC_SI
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['VC_SI_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['VC_SI_windw_list'].append(new_windw)               
            
        #17.TC_SI
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['TC_SI_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['TC_SI_windw_list'].append(new_windw)                

        #18.VC_KI
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['VC_KI_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['VC_KI_windw_list'].append(new_windw)    

        #19.TC_KI
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['TC_KI_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['TC_KI_windw_list'].append(new_windw)    

        #20.Mean_V
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['Mean_V_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['Mean_V_windw_list'].append(new_windw) 

        #21.Mean_T
        for windw_num in range(cycleNums-(window_size-1)):   #遍历每个窗口,对每个窗口
            new_windw=[]#缓存新窗口中的值
            for i in range(window_size):
                new_windw.append(ALL_feature_from_file[file_num]['Mean_T_list'][windw_num+i])#从第windw_num个元素开始，延续window_size个
            file_samples_windw['Mean_T_windw_list'].append(new_windw) 
        #-------------------至此，每个文件的各个特征的时间窗特征都已经计算完毕---------------- 
        
        #生成每个窗口所对应的RUL序列(Neol-Ni)/Neol *100
        rul_list=[]
        for cycle_i in range(cycleNums):    
            rul_list.append((cycleNums-cycle_i)/cycleNums*100)
        rul_list=rul_list[window_size-1:]  #获取每个窗口对应的RUL
        
        #构造特征矩阵
        #定义列表字典，每个特征的多个窗口的第i个值组合成一个列表，作为该列表的一个元素
        windw_index_feature_list_dict={'maxV_w_i':[],'TmaxV_w_i':[],'minV_w_i':[],'TminV_w_i':[],'maxT_w_i':[],
                                  'TmaxT_w_i':[],'minT_w_i':[],'TminT_w_i':[],'Cap_w_i':[],'VCE_w_i':[],
                                  'TCE_w_i':[],'VC_FI_w_i':[],'TC_FI_w_i':[],'VC_CI_w_i':[],'TC_CI_w_i':[],
                                  'VC_SI_w_i':[],'TC_SI_w_i':[],'VC_KI_w_i':[],'TC_KI_w_i':[],'Mean_V_w_i':[],
                                  'Mean_T_w_i':[]}
        

        for windw_index in range(window_size):  #对于每个窗口的第i个元素 windw_index_feature_list_dict['maxV_w_i'][index]对应特征的每个窗口的第i个元素组成的列表    
            windw_index_feature_list_dict['maxV_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  #遍历所有窗口file_samples_windw【'maxV_windw_list'】【【第一个窗】【第二个窗】】
                windw_index_feature_list_dict['maxV_w_i'][windw_index].append(file_samples_windw['maxV_windw_list'][windw_num][windw_index])
        for windw_index in range(window_size):      
            windw_index_feature_list_dict['TmaxV_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  
                windw_index_feature_list_dict['TmaxV_w_i'][windw_index].append(file_samples_windw['TmaxV_windw_list'][windw_num][windw_index])
        for windw_index in range(window_size): 
            windw_index_feature_list_dict['minV_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['minV_w_i'][windw_index].append(file_samples_windw['minV_windw_list'][windw_num][windw_index])
        for windw_index in range(window_size):  
            windw_index_feature_list_dict['TminV_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['TminV_w_i'][windw_index].append(file_samples_windw['TminV_windw_list'][windw_num][windw_index])
        for windw_index in range(window_size):  
            windw_index_feature_list_dict['maxT_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  
                windw_index_feature_list_dict['maxT_w_i'][windw_index].append(file_samples_windw['maxT_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):   
            windw_index_feature_list_dict['TmaxT_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['TmaxT_w_i'][windw_index].append(file_samples_windw['TmaxT_windw_list'][windw_num][windw_index])            
        for windw_index in range(window_size): 
            windw_index_feature_list_dict['minT_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['minT_w_i'][windw_index].append(file_samples_windw['minT_windw_list'][windw_num][windw_index])            
        for windw_index in range(window_size):    
            windw_index_feature_list_dict['TminT_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['TminT_w_i'][windw_index].append(file_samples_windw['TminT_windw_list'][windw_num][windw_index])            
        for windw_index in range(window_size):  
            windw_index_feature_list_dict['Cap_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  
                windw_index_feature_list_dict['Cap_w_i'][windw_index].append(file_samples_windw['Cap_windw_list'][windw_num][windw_index])
        for windw_index in range(window_size):   
            windw_index_feature_list_dict['VCE_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  
                windw_index_feature_list_dict['VCE_w_i'][windw_index].append(file_samples_windw['VCE_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):  
            windw_index_feature_list_dict['TCE_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  
                windw_index_feature_list_dict['TCE_w_i'][windw_index].append(file_samples_windw['TCE_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):     
            windw_index_feature_list_dict['VC_FI_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  
                windw_index_feature_list_dict['VC_FI_w_i'][windw_index].append(file_samples_windw['VC_FI_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):    
            windw_index_feature_list_dict['TC_FI_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['TC_FI_w_i'][windw_index].append(file_samples_windw['TC_FI_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):    
            windw_index_feature_list_dict['VC_CI_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['VC_CI_w_i'][windw_index].append(file_samples_windw['VC_CI_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):  
            windw_index_feature_list_dict['TC_CI_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['TC_CI_w_i'][windw_index].append(file_samples_windw['TC_CI_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):   
            windw_index_feature_list_dict['VC_SI_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['VC_SI_w_i'][windw_index].append(file_samples_windw['VC_SI_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):     
            windw_index_feature_list_dict['TC_SI_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  
                windw_index_feature_list_dict['TC_SI_w_i'][windw_index].append(file_samples_windw['TC_SI_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):  
            windw_index_feature_list_dict['VC_KI_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['VC_KI_w_i'][windw_index].append(file_samples_windw['VC_KI_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):  
            windw_index_feature_list_dict['TC_KI_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)): 
                windw_index_feature_list_dict['TC_KI_w_i'][windw_index].append(file_samples_windw['TC_KI_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):    
            windw_index_feature_list_dict['Mean_V_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  
                windw_index_feature_list_dict['Mean_V_w_i'][windw_index].append(file_samples_windw['Mean_V_windw_list'][windw_num][windw_index])                
        for windw_index in range(window_size):   
            windw_index_feature_list_dict['Mean_T_w_i'].append([])
            for windw_num in range(cycleNums-(window_size-1)):  
                windw_index_feature_list_dict['Mean_T_w_i'][windw_index].append(file_samples_windw['Mean_T_windw_list'][windw_num][windw_index])
                

                
        #定义临时列表 改所选特征在这里改----------------------------------------       
        temp_file_feature_matrix_list=[]
        for windw_index in range(window_size):
            temp_file_feature_matrix_list.append(windw_index_feature_list_dict['TminV_w_i'][windw_index])
#        for windw_index in range(window_size):
#            temp_file_feature_matrix_list.append(windw_index_feature_list_dict['TmaxT_w_i'][windw_index])
        for windw_index in range(window_size):
            temp_file_feature_matrix_list.append(windw_index_feature_list_dict['Cap_w_i'][windw_index])
        for windw_index in range(window_size):
            temp_file_feature_matrix_list.append(windw_index_feature_list_dict['VCE_w_i'][windw_index])
        for windw_index in range(window_size):
            temp_file_feature_matrix_list.append(windw_index_feature_list_dict['VC_FI_w_i'][windw_index])
#        for windw_index in range(window_size):
#            temp_file_feature_matrix_list.append(windw_index_feature_list_dict['TC_FI_w_i'][windw_index])
#        for windw_index in range(window_size):
#            temp_file_feature_matrix_list.append(windw_index_feature_list_dict['Mean_V_w_i'][windw_index])
        #在矩阵最后一列添加上RUL    
        temp_file_feature_matrix_list.append(rul_list)

        #形成特征矩阵
        file_feature_matrix=numpy.array(temp_file_feature_matrix_list)
        
        file_feature_matrix=file_feature_matrix.T   #转置，每一行对应一个放电窗口的特征+RUL
         
   
        
        
        
        #生成每个文件的所有窗口索引  [0,...195]
        windw_nums=cycleNums-(window_size-1)
        index_list=list(range(windw_nums))  
        #随机选出70%个索引作为测试集(从小到大)
        train_cycle_nums=int(windw_nums*0.7)
        train_cycle_index=sample(index_list,train_cycle_nums)
        train_cycle_index.sort()
        #使用剩余的产生剩余的索引作为测试集
        test_cycle_index=[]
        for cycle_i in range(windw_nums):
            if cycle_i not in train_cycle_index:
                test_cycle_index.append(cycle_i)
        test_cycle_index.sort()
            
         #2.产生训练集（特征窗口的70%）
        train_cycle_list=[] 
        for cycle_i in range(len(train_cycle_index)):
            train_cycle_list.append(file_feature_matrix[train_cycle_index[cycle_i]])
        #3.产生测试集（剩余的30%）
        test_cycle_list=[] 
        for cycle_i in range(len(test_cycle_index)):
            test_cycle_list.append(file_feature_matrix[test_cycle_index[cycle_i]])   
        
        #保存这个电池文件的train集合test集
        train_test_list.append(train_cycle_list)
        train_test_list.append(test_cycle_list)
        all_train_test_list.append(train_test_list)  
        
    return all_train_test_list        
    
        
#从这里开始设置但步长递增
def outRmseCompare(ALL_feature_from_file,size,method):
    """ 
    @Target:输出当前时间窗长度时的RMSE实验结果,可选择算法
    
    @Parameter:
        ALL_feature_from_file       #extract_feature_from_file函数残生的列表（包含特征）  
        size                        #设置时间窗的大小
        method                      #算法名
    @return
        计算生成的RMSE
    @Eg:
        RMSE=outRmseCompare(ALL_feature_from_file,5,'gbdt')    
    """ 

    all_train_test_list=extract_train_test_from_file_list_window(ALL_feature_from_file,size)       
     
    
    
    #%整合训练样本集和测试样本集-------------------------------------------------------------------------------------------------------------
    train_list=[]
    lest_list=[]
    
    Train_RUL=[]
    Test_RUL=[]
    #整合训练集到一个列表中train_list[样本数][特征索引]
    for i in range(len(all_train_test_list)):
        train_list.extend(all_train_test_list[i][0])
    #整合测试集到一个列表中lest_list[样本数][特征索引]
    for i in range(len(all_train_test_list)):
        lest_list.extend(all_train_test_list[i][1])    
    #------------------------------------------------------------------------------------------------
    #将训练集整合成矩阵    
    
    Train_X=numpy.array(train_list) 
    
    #将训练集按顺序排列，使拟合曲线更加平滑 ，使矩阵按某一列排序,为了方便，测试集也排序=======================
    arr=list(Train_X)
    u=len(arr[0])-1
    arr.sort(key=lambda x:x[u])
    Train_X=numpy.array(arr)
    
    
    #提取出RUL所在列
    Train_RUL=Train_X[:,len(Train_X[0])-1] 
    #提取出特征矩阵、不含RUL
    Train_X=Train_X[:,:len(Train_X[0])-1]    
       
    
    #将测试集整合成矩阵    
    Test_X=numpy.array(lest_list)  
    #将训练集按顺序排列，使拟合曲线更加平滑 ，使矩阵按某一列排序,为了方便，测试集也排序=======================
    arr=list(Test_X)
    u=len(arr[0])-1
    arr.sort(key=lambda x:x[u])
    Test_X=numpy.array(arr)
    
    
    #提取出RUL所在列
    Test_RUL=Test_X[:,len(Test_X[0])-1] 
    #提取出特征矩阵、不含RUL
    Test_X=Test_X[:,:len(Test_X[0])-1]       
    

#######################################标准化

#    Test_X=preprocessing.scale(Test_X)
#    Train_X=preprocessing.scale(Train_X)





    if method=='knn':
        knn = neighbors.KNeighborsRegressor()
        knn.fit(Train_X,Train_RUL)
        y_rbf=knn.predict(Test_X)  
               
    if method=='rf': 
        rf =ensemble.RandomForestRegressor(n_estimators=20)#随机森林
        rf.fit(Train_X, Train_RUL)
        y_rbf = rf.predict(Test_X) 
    
    if method=='svr': 
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        y_rbf = svr_rbf.fit(Train_X, Train_RUL).predict(Test_X)
        
    if method=='gbdt':    
        gbrt = ensemble.GradientBoostingRegressor() #GBRT
        gbrt.fit(Train_X, Train_RUL)
        y_rbf = gbrt.predict(Test_X) 
    
    if method=='neural_network':    
        nn=neural_network.MLPRegressor(hidden_layer_sizes=(7,),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)
        nn.fit(Train_X, Train_RUL)
        y_rbf = nn.predict(Test_X)     
    #%回归曲线----------------------------------------------------------------------------------------------------
    
    RMSE=rmse_calc(y_rbf,Test_RUL)
#    print('RMSE : ',RMSE)
#    precision_calc(y_rbf,Test_RUL)
    return RMSE

def multiTimeRMSE(ALL_feature_from_file,size,num):
    """
        @Target:重复批量地进行时间窗步长的测试，每次重新取样，设置时间窗为size，每个时间窗执行num次，取最小值或平均值
        
        @Parameter:
            ALL_feature_from_file       #extract_feature_from_file函数残生的列表（包含特征）  
            size                        #设置时间窗的大小
            num                         #每个时间窗执行num次
        @return
            计算生成的RMSE
        @Eg:
            RMSE=multiTimeRMSE(ALL_feature_from_file,5,10)    
    """
    #临时存放RMSE
    temp_rmse_list=[]
    #执行num次
    for i in range(num):
        temp_rmse_list.append(outRmse(ALL_feature_from_file,size))
    return numpy.array(temp_rmse_list).min()             #取最小还是取平均在这里设置
   

 ##############################################################################################
#开始实验    
#打开处理后的数据集源文件
#dataFileTrain = r'G:\pythonProject\discharge_db\B0034.dat'

db_file_list=[r'G:\pythonProject\discharge_db\B0036.dat',
#              r'G:\pythonProject\discharge_db\B0006.dat',         
#              r'G:\pythonProject\discharge_db\B0007.dat'
              ]
ALL_feature_from_file=extract_feature_from_file(db_file_list)
# knn rf svr gbdt
RMSE_dic={}

RMSE_dic['nn']=outRmseCompare(ALL_feature_from_file,1,'neural_network')  
RMSE_dic['rf']=outRmseCompare(ALL_feature_from_file,1,'rf')  
RMSE_dic['svr']=outRmseCompare(ALL_feature_from_file,1,'svr')     
RMSE_dic['gbdt_1']=outRmseCompare(ALL_feature_from_file,1,'gbdt')  
RMSE_dic['gbdt_30']=outRmseCompare(ALL_feature_from_file,30,'gbdt')  


print("RMSE_dic",RMSE_dic)



#temp=[]
#for i in range(10):
#    temp.append(outRmseCompare(ALL_feature_from_file,1,'neural_network'))
#
#print(min(temp))




#if __name__ == '__main__': main(r'G:\pythonProject\discharge_db\B0036.dat')



