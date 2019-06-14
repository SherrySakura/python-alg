# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:15:29 2018

@author: ZhengZhiyong

@Target: 特征提取函数

@Parameter:
       
"""
from copy import deepcopy
from math import sqrt
import pandas
import numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  #绘图
import shelve #小型数据库

def extract_feature(dcharge_cyc_list,output_feature_dict):    #特征提取的数量级单位应该是一个放电循环
    """ 
    @Target:提取特征 8 + 13个特征(其中两个是我自己加的)
    
    @Parameter:
        dcharge_cyc_list                 #经过discharge_dataset_format.py处理的数据库中的'discharge'的第几个循环,db['discharge'][cycle_num]->['Voltage_measured']
        output_feature_dict              #函数输出的特征值组成的字典,如output_feature_dict['maxV']
    
    @Eg:
        feature_dict={}
        extract_feature(temp_db_list,feature_dict)
        plt.plot(feature_dict['VC_FI']) 
        
    db['discharge'][cycle_num]['Voltage_measured']
    """
    #提取电压曲线特征--------------------------
    #1.最大电压list（maxV）
    #2.最大电压所对应的时间list（TmaxV）
    #3.最大电压list（minV）
    #4.最大电压所对应的时间list（TminV）
    #不定义也可以，列出来方便看
    maxV=0
    TmaxV=0
    minV=0
    TminV=0
    
    #提取温度曲线特征--------------------------
    #1.最大温度list（maxT）
    #2.最大温度所对应的时间list（TmaxT）
    #3.最小温度list（minT） 
    #4.最小温度所对应的时间list（TminT）
    maxT=0
    TmaxT=0
    minT=0
    TminT=0
    
    #其他特征-----------------
    #1.电池容量(Cap)
    #2.电压信号能量(VCE)
    #3.温度信号能量(TCE)
    #4.电压曲线信号波动指数(VC_FI)
    #5.温度曲线信号波动指数(TC_FI)
    #6.电压曲线的曲率指数(VC_CI)
    #7.温度曲线的曲率指数(TC_CI)
    #8.
    #9.
    #10.电压信号的偏度指数(VC_SI)
    #11.温度信号的偏度指数(TC_SI)
    #12.电压信号的峰度指数(VC_KI)
    #13.温度信号的峰度指数(TC_KI)
    Cap=0
    VCE=0
    TCE=0
    VC_FI=0
    TC_FI=0
    VC_CI=0
    TC_CI=0
    #留着
    
    VC_SI=0
    TC_SI=0
    VC_KI=0
    TC_KI=0
    
    #自己加的特征
    #1.电压曲线均值
    Mean_V=0
    #2.温度曲线均值
    Mean_T=0
    
    #电压曲线特征----------------4个基本特征----------------------
    maxVolt=max(dcharge_cyc_list['Voltage_measured'])         #获取所有的volT最大值
    maxV=maxVolt
    
    index=dcharge_cyc_list['Voltage_measured'].index(maxVolt) #获取最大电压所对应的时间
    TmaxVolt=dcharge_cyc_list['Time'][index]
    TmaxV=TmaxVolt
    
    minVolt=min(dcharge_cyc_list['Voltage_measured'])         #获取所有的volT最小值
    minV=minVolt
    
    index=dcharge_cyc_list['Voltage_measured'].index(minVolt) #获取最小电压所对应的时间
    TminVolt=dcharge_cyc_list['Time'][index]
#    TminV=TminVolt
    TminV=TminVolt/1000
    
    #电压曲线特征----------------4个基本特征----------------------
    maxTemp=max(dcharge_cyc_list['Temperature_measured'])      #获取所有的temperature最大值
    maxT=maxTemp
    
    index=dcharge_cyc_list['Temperature_measured'].index(maxTemp) #获取最大温度所对应的时间
    TmaxTemp=dcharge_cyc_list['Time'][index]
#    TmaxT=TmaxTemp
    TmaxT=TmaxTemp/1000
    
    minTemp=min(dcharge_cyc_list['Temperature_measured'])      #获取所有的temperature最小值
    minT=minTemp
    
    index=dcharge_cyc_list['Temperature_measured'].index(minTemp) #获取最小温度所对应的时间
    TminTemp=dcharge_cyc_list['Time'][index]
    TminT=TminTemp
    
    #其他13个特征
    #1.Cap
    capsum=0
    for arrayNum in range(len(dcharge_cyc_list['Current_measured'])):   #计算整个电流波形的积分
        timeDif = dcharge_cyc_list['Time'][arrayNum+1]-dcharge_cyc_list['Time'][arrayNum] if arrayNum<len(dcharge_cyc_list['Time'])-1 else 0
        capsum+=dcharge_cyc_list['Current_measured'][arrayNum]*timeDif
#    Cap=-1*capsum
    Cap=-1*capsum/1000

    #2.电压信号能量(VCE)
    energysum_v=0
    for arrayNum in range(len(dcharge_cyc_list['Voltage_measured'])):   #计算电压信号能量
        timeDif = dcharge_cyc_list['Time'][arrayNum+1]-dcharge_cyc_list['Time'][arrayNum] if arrayNum<len(dcharge_cyc_list['Time'])-1 else 0
        energysum_v+=abs(dcharge_cyc_list['Voltage_measured'][arrayNum])**2*timeDif
#    VCE=energysum_v
    VCE=energysum_v/10000
    
    #3.温度信号能量(TCE)
    energysum_t=0
    for arrayNum in range(len(dcharge_cyc_list['Temperature_measured'])):   #计算温度信号能量
        timeDif = dcharge_cyc_list['Time'][arrayNum+1]-dcharge_cyc_list['Time'][arrayNum] if arrayNum<len(dcharge_cyc_list['Time'])-1 else 0
        energysum_t+=abs(dcharge_cyc_list['Temperature_measured'][arrayNum])**2*timeDif
    TCE=energysum_t
    
    #4.电压曲线信号波动指数(VC_FI)和电压曲线平均值(Mean_V)
    mean_v=0   #电压曲线平均值  
    for arrayNum in range(len(dcharge_cyc_list['Voltage_measured'])):
        mean_v+=dcharge_cyc_list['Voltage_measured'][arrayNum]
    mean_v=mean_v/len(dcharge_cyc_list['Voltage_measured'])    #计算平均值
    Mean_V=mean_v
    
    FI_V=0
    Period=(numpy.array(dcharge_cyc_list['Time'][1:])-numpy.array(dcharge_cyc_list['Time'][:len(dcharge_cyc_list['Time'])-1])).mean()
    for arrayNum in range(len(dcharge_cyc_list['Voltage_measured'])): #计算每个循环的FI
        FI_V+=(dcharge_cyc_list['Voltage_measured'][arrayNum]-mean_v)**2
    FI_V=sqrt(FI_V)*Period
#    VC_FI=FI_V*10000
    VC_FI=FI_V/10
    
    #5.温度曲线信号波动指数(TC_FI)和温度曲线均值(Mean_T)
    mean_t=0
    for arrayNum in range(len(dcharge_cyc_list['Temperature_measured'])):
        mean_t+=dcharge_cyc_list['Temperature_measured'][arrayNum]
    mean_t=mean_t/len(dcharge_cyc_list['Temperature_measured'])    #计算平均值
    Mean_T=mean_t
    
    FI_T=0
    Period=(numpy.array(dcharge_cyc_list['Time'][1:])-numpy.array(dcharge_cyc_list['Time'][:len(dcharge_cyc_list['Time'])-1])).mean()
    for arrayNum in range(len(dcharge_cyc_list['Temperature_measured'])):
        FI_T+=(dcharge_cyc_list['Temperature_measured'][arrayNum]-mean_t)**2
    FI_T=sqrt(FI_T)*Period
#    TC_FI=FI_T
    TC_FI=FI_T/100
    
    #6.电压曲线的曲率指数(VC_CI)
    Ser_Volt=pandas.Series(dcharge_cyc_list['Voltage_measured'])    #电压序列
    Ser_Time=pandas.Series(dcharge_cyc_list['Time'])                #时间序列
   
    DI1_Volt=Ser_Volt.diff()     #计算电压序列一阶差
    DI1_Volt=DI1_Volt.dropna()   #删掉第一个NaN  len值减一  DI1_Volt为Serise类型
    DT1_Time=Ser_Time.diff()     #计算时间序列一阶差
    DT1_Time=DT1_Time.dropna()   #删掉第一个NaN  len值减一  DI1_Time为Serise类型
    DIDT1_Volt=DI1_Volt/DT1_Time  #计算电压的一阶导数di/dt  DIDT1_Volt为Serise类型
    
    DI2_Volt=DI1_Volt.diff()     #计算电压序列二阶差
    DI2_Volt=DI2_Volt.dropna()   #删掉第一个NaN  len值减一  DI2_Volt为Serise类型
    #DI2_Volt首位补上一个0
    DI2_Volt=list(DI2_Volt)
    DI2_Volt.insert(0,0)
    DI2_Volt=pandas.Series(DI2_Volt)
    #DT1_Time重新排序索引，不然计算时候匹配不上
    DT1_Time=list(DT1_Time)   
    DT1_Time=pandas.Series(DT1_Time)
    
    DIDT2_Volt=DI2_Volt/DT1_Time #计算电压的二阶导数di/dt  DIDT2_Volt为Serise类型  
    
    #序列元素长度补齐，并重新索引
    DIDT1_Volt=list(DIDT1_Volt)   #补一个（补满）
    DIDT1_Volt.insert(0,0)
    DIDT1_Volt=pandas.Series(DIDT1_Volt)
    
    DIDT2_Volt=list(DIDT2_Volt)   #补一个（补满）
    DIDT2_Volt.insert(0,0)
    DIDT2_Volt=pandas.Series(DIDT2_Volt)

    CI_Volt=(DIDT2_Volt/((1+DIDT1_Volt**2)**1.5)).sum()/len(dcharge_cyc_list['Time'])  
    VC_CI=CI_Volt
    
    #7.温度曲线的曲率指数(TC_CI)----------------------------------------------
    Ser_Temp=pandas.Series(dcharge_cyc_list['Temperature_measured'])    #温度序列
    Ser_Time=pandas.Series(dcharge_cyc_list['Time'])                    #时间序列
   
    DI1_Temp=Ser_Temp.diff()     #计算温度序列一阶差
    DI1_Temp=DI1_Temp.dropna()   #删掉第一个NaN  len值减一  DI1_Temp为Serise类型
    DT1_Time=Ser_Time.diff()     #计算时间序列一阶差
    DT1_Time=DT1_Time.dropna()   #删掉第一个NaN  len值减一  DI1_Time为Serise类型
    DIDT1_Temp=DI1_Temp/DT1_Time  #计算温度的一阶导数di/dt  DIDT1_Temp为Serise类型
    
    DI2_Temp=DI1_Temp.diff()     #计算温度序列二阶差
    DI2_Temp=DI2_Temp.dropna()   #删掉第一个NaN  len值减一  DI2_Temp为Serise类型
    #DI2_Temp首位补上一个0
    DI2_Temp=list(DI2_Temp)
    DI2_Temp.insert(0,0)
    DI2_Temp=pandas.Series(DI2_Temp)
    #DT1_Time重新排序索引，不然计算时候匹配不上
    DT1_Time=list(DT1_Time)   
    DT1_Time=pandas.Series(DT1_Time)
    
    DIDT2_Temp=DI2_Temp/DT1_Time #计算温度的二阶导数di/dt  DIDT2_Temp为Serise类型  
    
    #序列元素长度补齐，并重新索引
    DIDT1_Temp=list(DIDT1_Temp)   #补一个（补满）
    DIDT1_Temp.insert(0,0)
    DIDT1_Temp=pandas.Series(DIDT1_Temp)
    
    DIDT2_Temp=list(DIDT2_Temp)   #补一个（补满）
    DIDT2_Temp.insert(0,0)
    DIDT2_Temp=pandas.Series(DIDT2_Temp)

    CI_Temp=(DIDT2_Temp/((1+DIDT1_Temp**2)**1.5)).sum()/len(dcharge_cyc_list['Time'])  
    TC_CI=CI_Temp   
    
    #8.
    #9.
    
    #10.电压信号的偏度指数(VC_SI)
    Ser_Volt=pandas.Series(dcharge_cyc_list['Voltage_measured'])         #电压序列
    VC_SI=Ser_Volt.skew()
    
    #11.温度信号的偏度指数(TC_SI)
    Ser_Temp=pandas.Series(dcharge_cyc_list['Temperature_measured'])     #温度序列
    TC_SI=Ser_Temp.skew()  
    
    #12.电压信号的峰度指数(VC_KI)
    Ser_Volt=pandas.Series(dcharge_cyc_list['Voltage_measured'])         #电压序列
    VC_KI=Ser_Volt.kurt()
    
    #13.温度信号的峰度指数(TC_KI)
    Ser_Temp=pandas.Series(dcharge_cyc_list['Temperature_measured'])     #温度序列
    TC_KI=Ser_Temp.kurt()   
    
    #print('cycle num : ',i)
    
    output_feature_dict['maxV']=maxV
    output_feature_dict['TmaxV']=TmaxV
    output_feature_dict['minV']=minV    
    output_feature_dict['TminV']=TminV
    output_feature_dict['maxT']=maxT
    output_feature_dict['TmaxT']=TmaxT
    output_feature_dict['minT']=minT
    output_feature_dict['TminT']=TminT
    output_feature_dict['Cap']=Cap       
    output_feature_dict['VCE']=VCE
    output_feature_dict['TCE']=TCE   
    output_feature_dict['VC_FI']=VC_FI  #和瑞姐一样，和原文不一样
    output_feature_dict['TC_FI']=TC_FI  #和瑞姐一样，和原文不一样     
    output_feature_dict['VC_CI']=VC_CI  #和原文不一样 带考究
    output_feature_dict['TC_CI']=TC_CI  #和原文不一样 带考究
    output_feature_dict['VC_SI']=VC_SI        
    output_feature_dict['TC_SI']=TC_SI
    output_feature_dict['VC_KI']=VC_KI
    output_feature_dict['TC_KI']=TC_KI
    output_feature_dict['Mean_V']=Mean_V  
    output_feature_dict['Mean_T']=Mean_T

def main():
    print('extract_feature model load success! By__ZZY')


def pca_down(train_x,input_x,n):
    """ 
    @Target:训练pca并转换输入的数据
    
    @Parameter:
        train_x       #PCA模型的训练集
        input_x       #带转换的数据
        n             #需要输出的维度
    @return
        input_x经过pca转换后的数据
    @Eg:
        NEW_X=pca_down(X,X,2)
    """
    #plt.scatter(train_x[:, 0], train_x[:, 1],marker='o')
    print(train_x.shape)
    #print(train_x)
    
    pca = PCA(n_components=n)
    pca.fit(train_x)
    print('explained_variance_ratio_: ',pca.explained_variance_ratio_)# 返回各个成分各自的方差百分比(贡献率)
    print('explained_variance_: ',pca.explained_variance_)# 返回模型的各个特征值
    print('components_ : ',pca.components_ )     # 返回模型的各个特征向量
    print('get_covariance() : ',pca.get_covariance() )     # 计算与生成模型的数据协方差
    #print('singular_values_ : ',pca.singular_values_ )     # 奇异值对应于每个选定的分量
    
    return pca.transform(input_x)
    #output_x.extend(pca.transform(train_x))   #使用训练好的PCA转换新数据
    #plt.plot(output_x) 
    #plt.scatter(output_x[:, 0], output_x[:, 1],marker='o')
    
def rmse_calc(predict_list,real_list):
    """ 
    @Target:计算测试结果与真实结果的根均方误差
    
    @Parameter:
        predict_list       #预测得到的结果   [1,2,3] 或者 array([1,2,3])
        real_list          #真实数据

    @return
        计算得到的rmse
    @Eg:
        RMSE=rmse_calc(predict_list,real_list)
    """
    temp_list=[]
    for i in range(len(predict_list)):
        temp_list.append((predict_list[i]-real_list[i])**2)
    return numpy.sqrt(numpy.sum(temp_list)/len(predict_list))




def mean_sudden_filter(temp_list,windown_size,multiple):
    """ 
    @Target:基于滑动窗体均值和异常点突变程度的滤波算法，对窗体求均值，如果下一个点的值超过均值的multiple倍，则将其替换为窗体的最后一个值
    
    @Parameter:
        temp_list             #待滤波的列表
        windown_size          #窗体尺寸
        multiple              #异常阈值（相对于窗体均值的倍数）
        
    @return
        滤波后的列表
    @Eg:
        a=mean_sudden_filter(a,5,3)
    """  
    min=numpy.min(temp_list)
    temp_list=list(numpy.array(temp_list)-min)  #将整个列表抬高 ，主要是为了防止小于0的值  
    
    
    
    window=[]
    window=temp_list[:windown_size] #初始化窗体

    for i in range(len(temp_list)-windown_size):
        mean=numpy.mean(window)  #求窗体均值
        if temp_list[i+windown_size]>mean*multiple:    #异常点
            temp_list[i+windown_size]=window[windown_size-1] #替换异常点
            window=temp_list[i:i+windown_size]  #更新窗体
        else:
            window=temp_list[i:i+windown_size]  #更新窗体
            
    temp_list=list(numpy.array(temp_list)+min)  #将整个列表抬高 ，主要是为了防止小于0的值  
    return temp_list   
   

     
def smooth(a,WSZ):
    """ 
    @Target:基于滑动窗体均值和异常点突变程度的滤波算法，对窗体求均值，如果下一个点的值超过均值的multiple倍，则将其替换为窗体的最后一个值
    
    @Parameter:
        a:         #原始数据，NumPy 1-D array containing the data to be smoothed,必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
        WSZ:       #smoothing window size needs, which must be odd number

    @return
        滤波后的列表
    @Eg:
        Train_X=smooth(Train_X,1)
    """  

    out0 = numpy.convolve(a,numpy.ones(WSZ,dtype=int),'valid')/WSZ
    r = numpy.arange(1,WSZ-1,2)
    start = numpy.cumsum(a[:WSZ-1])[::2]/r
    stop = (numpy.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return numpy.concatenate((  start , out0, stop  ))
    

def precision_calc(predict_list,real_list):
    """ 
    @Target:计算并输出预测结果的准确度
    
    @Parameter:
        predict_list       #预测得到的结果   [1,2,3] 或者 array([1,2,3])
        real_list          #真实数据

    @return
        打印精度分析结果：
            1.predict低于real95%的个数、百分比
            2.predict高于real95%的个数、百分比
    @Eg:
        precision_calc(b,a)
    """  
    NUM_T105=0 #大于real值95%的个数
    NUM_D95=0
    NUM_COMMON=0

    for i in range(len(predict_list)):
        if predict_list[i]>real_list[i]+5:
            NUM_T105+=1
        elif predict_list[i]<real_list[i]-5:
            NUM_D95+=1
        else:
            NUM_COMMON+=1
    print('precision : ',NUM_COMMON/len(predict_list)*100,'%')
#    REE_RATE=0
#    for i in range(len(predict_list)):
#        REE_RATE=real_list[i]*0.05
#        if predict_list[i]>real_list[i]+REE_RATE:
#            NUM_T105+=1
#        elif predict_list[i]<real_list[i]-REE_RATE:
#            NUM_D95+=1
#        else:
#            NUM_COMMON+=1   
#    print('precision : ',NUM_COMMON/len(predict_list)*100,'%')
        
    #按照参考论文， error%=（1-real/predict）*100
#    PRECISION_SUM=0
#    for i in range(len(predict_list)):
#        PRECISION_SUM+=100-(1-abs(real_list[i]/predict_list[i]))*100    #有除零风险
#    print('precision : ',PRECISION_SUM/len(predict_list),' %')

#    ERROR_SUM=0
#    for i in range(len(predict_list)):
#        ERROR_SUM+=abs((predict_list[i]-real_list[i]))/predict_list[i]*100    #有除零风险
#    print('precision : ',100-ERROR_SUM/len(predict_list),'%')


    print('NUM_T105 : ',NUM_T105)
    print('NUM_D95 : ',NUM_D95)
    print('NUM_COMMON : ',NUM_COMMON)

 

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
        battary_feature_dict_list.append(battary_feature_dict)   #battary_feature_dict['maxV_list']=maxV_list[]   数据结构
   
    return battary_feature_dict_list



#E:\Anaconda3\Lib\site-packages    
if __name__ == '__main__': main()



