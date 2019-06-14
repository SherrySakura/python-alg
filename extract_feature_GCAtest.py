# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:15:29 2018

@author: ZhengZhiyong

@Target:提取特征 8 + 13个特征  (B0005)

@Parameter:
    dataBaseFile = r'G:\pythonProject\discharge_db\B0034.dat'
    db['discharge'][cycle_num]['Voltage_measured']
    
"""


from copy import deepcopy
import shelve #小型数据库
import matplotlib.pyplot as plt  #绘图
from math import sqrt
import pandas
import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import preprocessing  #标准化处理
from zzy_model import extract_feature#导入自定义特征提取模块
from zzy_model import pca_down    
from zzy_model import rmse_calc
from zzy_model import mean_sudden_filter
from zzy_model import smooth



#出组图
#plt.subplot(1, 2, 1) # （行，列，活跃区）
#plt.plot([1.5,3.5,-2,1.6])  
#plt.subplot(1, 2, 2)
#plt.plot([1.5,3.5,-2,1.6])  
#plt.show()
#
#plt.subplot(1, 2, 1) # （行，列，活跃区）
#plt.plot([1.5,3.5,-2,1.6])  
#plt.subplot(1, 2, 2)
#plt.plot([1.5,3.5,-2,1.6])  
#plt.show()


#打开处理后的数据集源文件

dataFileTrain = r'G:\pythonProject\discharge_db\B0033.dat'


db = shelve.open(dataFileTrain, flag='c', protocol=None, writeback=False)

#定义临时列表，缓存
temp_db_list=deepcopy(db['discharge'])
db.close()      
#提取电压曲线特征--------------------------
#1.最大电压list（maxV）
#2.最大电压所对应的时间list（TmaxV）
#3.最大电压list（minV）
#4.最大电压所对应的时间list（TminV）
maxV=[]
TmaxV=[]
minV=[]
TminV=[]

#提取温度曲线特征--------------------------
#1.最大温度list（maxT）
#2.最大温度所对应的时间list（TmaxT）
#3.最小温度list（minT） 
#4.最小温度所对应的时间list（TminT）
maxT=[]
TmaxT=[]
minT=[]
TminT=[]

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
Cap=[]
VCE=[]
TCE=[]
VC_FI=[]
TC_FI=[]
VC_CI=[]
TC_CI=[]
#留着

VC_SI=[]
TC_SI=[]
VC_KI=[]
TC_KI=[]

#自己加的特征
#1.电压曲线均值
Mean_V=[]
#2.温度曲线均值
Mean_T=[]


for i in range(len(temp_db_list)):   #迭代所有循环、

    #电压曲线特征----------------4个基本特征----------------------
    maxVolt=max(temp_db_list[i]['Voltage_measured'])         #获取所有的volT最大值
    maxV.append(maxVolt)
    
    index=temp_db_list[i]['Voltage_measured'].index(maxVolt) #获取最大电压所对应的时间
    TmaxVolt=temp_db_list[i]['Time'][index]
    TmaxV.append(TmaxVolt)
    
    minVolt=min(temp_db_list[i]['Voltage_measured'])         #获取所有的volT最小值
    minV.append(minVolt)
    
    index=temp_db_list[i]['Voltage_measured'].index(minVolt) #获取最小电压所对应的时间
    TminVolt=temp_db_list[i]['Time'][index]
#    TminV.append(TminVolt)
    TminV.append(TminVolt/1000)
    
    #电压曲线特征----------------4个基本特征----------------------
    maxTemp=max(temp_db_list[i]['Temperature_measured'])      #获取所有的temperature最大值
    maxT.append(maxTemp)
    
    index=temp_db_list[i]['Temperature_measured'].index(maxTemp) #获取最大温度所对应的时间
    TmaxTemp=temp_db_list[i]['Time'][index]
#    TmaxT.append(TmaxTemp)
    TmaxT.append(TmaxTemp/1000)
    
    minTemp=min(temp_db_list[i]['Temperature_measured'])      #获取所有的temperature最小值
    minT.append(minTemp)
    
    index=temp_db_list[i]['Temperature_measured'].index(minTemp) #获取最小温度所对应的时间
    TminTemp=temp_db_list[i]['Time'][index]
    TminT.append(TminTemp)
    
    #其他13个特征
    #1.Cap
    capsum=0
    for arrayNum in range(len(temp_db_list[i]['Current_measured'])):   #计算整个电流波形的积分
        timeDif = temp_db_list[i]['Time'][arrayNum+1]-temp_db_list[i]['Time'][arrayNum] if arrayNum<len(temp_db_list[i]['Time'])-1 else 0
        capsum+=temp_db_list[i]['Current_measured'][arrayNum]*timeDif
#    Cap.append(-1*capsum)
    Cap.append(-1*capsum/2400)
    #2.电压信号能量(VCE)
    energysum_v=0
    for arrayNum in range(len(temp_db_list[i]['Voltage_measured'])):   #计算电压信号能量
        timeDif = temp_db_list[i]['Time'][arrayNum+1]-temp_db_list[i]['Time'][arrayNum] if arrayNum<len(temp_db_list[i]['Time'])-1 else 0
        energysum_v+=abs(temp_db_list[i]['Voltage_measured'][arrayNum])**2*timeDif
#    VCE.append(energysum_v)
    VCE.append(energysum_v/10000)
    #3.温度信号能量(TCE)
    energysum_t=0
    for arrayNum in range(len(temp_db_list[i]['Temperature_measured'])):   #计算温度信号能量
        timeDif = temp_db_list[i]['Time'][arrayNum+1]-temp_db_list[i]['Time'][arrayNum] if arrayNum<len(temp_db_list[i]['Time'])-1 else 0
        energysum_t+=abs(temp_db_list[i]['Temperature_measured'][arrayNum])**2*timeDif
    TCE.append(energysum_t)
    
    #4.电压曲线信号波动指数(VC_FI)和电压曲线平均值(Mean_V)
    mean_v=0   #电压曲线平均值  
    for arrayNum in range(len(temp_db_list[i]['Voltage_measured'])):
        mean_v+=temp_db_list[i]['Voltage_measured'][arrayNum]
    mean_v=mean_v/len(temp_db_list[i]['Voltage_measured'])    #计算平均值
    Mean_V.append(mean_v)
    

    FI_V=0
    Period=(numpy.array(temp_db_list[i]['Time'][1:])-numpy.array(temp_db_list[i]['Time'][:len(temp_db_list[i]['Time'])-1])).mean()
    
    for arrayNum in range(len(temp_db_list[i]['Voltage_measured'])): #计算每个循环的FI
        FI_V+=(temp_db_list[i]['Voltage_measured'][arrayNum]-mean_v)**2
    FI_V=sqrt(FI_V)*Period
    #VC_FI.append(FI_V*10000)
    VC_FI.append(FI_V/24)
    
    
    #5.温度曲线信号波动指数(TC_FI)和温度曲线均值(Mean_T)
    mean_t=0
    for arrayNum in range(len(temp_db_list[i]['Temperature_measured'])):
        mean_t+=temp_db_list[i]['Temperature_measured'][arrayNum]
    mean_t=mean_t/len(temp_db_list[i]['Temperature_measured'])    #计算平均值
    Mean_T.append(mean_t)
    
    FI_T=0
    Period=(numpy.array(temp_db_list[i]['Time'][1:])-numpy.array(temp_db_list[i]['Time'][:len(temp_db_list[i]['Time'])-1])).mean()
    for arrayNum in range(len(temp_db_list[i]['Temperature_measured'])):
        FI_T+=(temp_db_list[i]['Temperature_measured'][arrayNum]-mean_t)**2
    FI_T=sqrt(FI_T)*Period
#    TC_FI.append(FI_T)
    TC_FI.append(FI_T/100)
    
    #6.电压曲线的曲率指数(VC_CI)
    Ser_Volt=pandas.Series(temp_db_list[i]['Voltage_measured'])    #电压序列
    Ser_Time=pandas.Series(temp_db_list[i]['Time'])                #时间序列
   
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

    CI_Volt=(DIDT2_Volt/((1+DIDT1_Volt**2)**1.5)).sum()/len(temp_db_list[i]['Time'])  
    VC_CI.append(CI_Volt)
    
    #7.温度曲线的曲率指数(TC_CI)----------------------------------------------
    Ser_Temp=pandas.Series(temp_db_list[i]['Temperature_measured'])    #温度序列
    Ser_Time=pandas.Series(temp_db_list[i]['Time'])                    #时间序列
   
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

    CI_Temp=(DIDT2_Temp/((1+DIDT1_Temp**2)**1.5)).sum()/len(temp_db_list[i]['Time'])  
    TC_CI.append(CI_Temp)    
    
    #8.
    #9.
    
    #10.电压信号的偏度指数(VC_SI)
    Ser_Volt=pandas.Series(temp_db_list[i]['Voltage_measured'])         #电压序列
    VC_SI.append(Ser_Volt.skew())
    
    #11.温度信号的偏度指数(TC_SI)
    Ser_Temp=pandas.Series(temp_db_list[i]['Temperature_measured'])     #温度序列
    TC_SI.append(Ser_Temp.skew())    
    
    #12.电压信号的峰度指数(VC_KI)
    Ser_Volt=pandas.Series(temp_db_list[i]['Voltage_measured'])         #电压序列
    VC_KI.append(Ser_Volt.kurt())
    
    #13.温度信号的峰度指数(TC_KI)
    Ser_Temp=pandas.Series(temp_db_list[i]['Temperature_measured'])     #温度序列
    TC_KI.append(Ser_Temp.kurt())   
    
    #print('cycle num : ',i)


#1出原始特征图
VCE=VCE[1:len(VCE)]
VC_FI=VC_FI[1:len(VC_FI)]
Cap=Cap[1:len(Cap)]
VC_FI[0]=2.55#############################################################################################################################
VCE[0]=3.55
x=list(range(len(VCE)))
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(x,VCE,'-x',linewidth=3,label='VCE')      #6500           /1000      = 6.5---------------------------------
plt.plot(x,VC_FI, '-o',linewidth=3,label='VFI')     #40             /10            =4--------------------------------
plt.plot(x,Cap, '-o',linewidth=3,label='Cap')  

#plt.grid(True)
plt.legend(loc='upper right')
plt.xlabel('Discharge cycle') # 给 x 轴添加标签
plt.ylabel('Value') # 给 y 轴添加标签
#plt.show()




#%特征数据预处理-------------------------------------------------------------------------------------------------------------
def feature_preprocessing1(temp_feature_list):  #VCE
    #去除第一个点
   # temp_feature_list=temp_feature_list[3:]
    #移动平均l滤波
#    temp_feature_list=smooth(temp_feature_list,5)
    #标准化  然后抬到0以上   乘以一定倍数
#    temp_feature_list=preprocessing.scale(temp_feature_list)
#    temp_feature_list=list(numpy.array(temp_feature_list)*5000)
#    temp_feature_list=list(numpy.array(temp_feature_list))
    #均值突变滤波
    temp_feature_list=mean_sudden_filter(temp_feature_list,10,2.2)   
    
    return temp_feature_list

def feature_preprocessing2(temp_feature_list):   #VFI
    #去除第一个点
  #  temp_feature_list=temp_feature_list[3:]
    #移动平均l滤波
#    temp_feature_list=smooth(temp_feature_list,5)
    #标准化  然后抬到0以上   乘以一定倍数
#    temp_feature_list=preprocessing.scale(temp_feature_list)
#    temp_feature_list=list(numpy.array(temp_feature_list)*5000)
#    temp_feature_list=list(numpy.array(temp_feature_list))
    #均值突变滤波
    temp_feature_list=mean_sudden_filter(temp_feature_list,10,2.5)   
    
    return temp_feature_list

#对输入列表进行无量纲化处理
def Nondimensionalization(temp_feature_list):
    minE = min(temp_feature_list)
    maxE = max(temp_feature_list)
    for index in range(len(temp_feature_list)):
        temp_feature_list[index]-=minE
        temp_feature_list[index]/=(maxE-minE)
    return temp_feature_list
    


#2出滤波后的图


x=list(range(len(VCE)))
plt.figure(figsize=(8, 5.3), dpi=80)
plt.plot(x,VCE,'-x',dashes=[6, 2],color='darkgreen',ms=8.5,linewidth=1.5,label='raw VCE')      #6500           /1000      = 6.5---------------------------------
plt.plot(x,VC_FI, '-o',dashes=[6, 2],color='indianred',ms=8.5,linewidth=1.5,label='raw VFI')

VCE=feature_preprocessing1(VCE)
VC_FI=feature_preprocessing2(VC_FI)

plt.plot(x,VCE,'-x',linewidth=1.5,color='#1f77b4',ms=8.5,label='filtered VCE')      #6500           /1000      = 6.5---------------------------------
plt.plot(x,VC_FI, '-o',linewidth=1.5,color='orange',ms=8.5,label='filtered VFI')     #40             /10            =4--------------------------------
#plt.grid(True)

#3个标签大小
font1 = {'size': 14}
#刻度大小
plt.tick_params(labelsize=14)

plt.legend(loc='upper right',prop=font1)
plt.xlabel('Discharge cycle',font1) # 给 x 轴添加标签
plt.ylabel('Feature Value',font1) # 给 y 轴添加标签
plt.rcParams['axes.linewidth'] = 1.2
#加(a)标签
plt.text(-30, 5.95, '(a)', fontsize=18)
plt.show()
#resultpath=r'C:\Users\ZhengZhiyong\Desktop\锂电池rul论文\电池图片\21.pdf'
#plt.savefig(resultpath)
# --------------------------------------------------------------

#3.计算关联度
#1.构建参考列
rul_list=[]
for i in range(len(VC_FI)):
    rul_list.append(i)
rul_list_up=deepcopy(rul_list)
rul_list.reverse()
rul_list_down=deepcopy(rul_list)
#2.无量纲化并绘图
rul_list_up = Nondimensionalization(rul_list_up)
rul_list_down = Nondimensionalization(rul_list_down)
VCE = Nondimensionalization(VCE)
VC_FI = Nondimensionalization(VC_FI)


plt.figure(figsize=(8, 5.3), dpi=80)
plt.plot(x,VCE, '-o',linewidth=1.5,color='#1f77b4',ms=8.5,label='filtered VCE')     #40             /10            =4--------------------------------
plt.plot(x,rul_list_down, '-',linewidth=4.5,dashes=[6, 2],color='gray',ms=8.5,label='reference line') 
#标签大小
font1 = {'size': 14}
#刻度大小
plt.tick_params(labelsize=14)

plt.legend(loc='upper right',prop=font1)
plt.xlabel('Discharge cycle',font1) # 给 x 轴添加标签
plt.ylabel('Feature Value',font1) # 给 y 轴添加标签
#加(a)标签
plt.text(-35, 1.02, '(b)', fontsize=18)
plt.show()
#resultpath=r'C:\Users\ZhengZhiyong\Desktop\锂电池rul论文\电池图片\22.pdf'
#plt.savefig(resultpath)


plt.figure(figsize=(8, 5.3), dpi=80)
plt.plot(x,VC_FI, '-x',linewidth=1.5,color='orange',ms=8.5,label='filtered VFI') 
plt.plot(x,rul_list_up,'-',linewidth=4.5,dashes=[6, 2],color='gray',ms=8.5,label='reference line')

#plt.grid(True)
#标签大小
font1 = {'size': 14}
#刻度大小
plt.tick_params(labelsize=14)

plt.legend(loc='upper right',prop=font1)
plt.xlabel('Discharge cycle',font1) # 给 x 轴添加标签
plt.ylabel('Feature Value',font1) # 给 y 轴添加标签
#加(a)标签
plt.text(-35, 1.02, '(c)', fontsize=18)
plt.show()
#resultpath=r'C:\Users\ZhengZhiyong\Desktop\锂电池rul论文\电池图片\23.pdf'
#plt.savefig(resultpath)
#计算


a=[1,1,1]
b=[[1,2,3],[2,2,2]]


def GCA_analysis(ref_list,test_lists):
    test_num=len(test_lists)#待比较的数列数量
    list_len=len(ref_list)  #每个数列的长度
    X=numpy.zeros([test_num,list_len])
    for i in range(test_num):
        for j in range(list_len):
            X[i][j]=abs(test_lists[i][j]-ref_list[j])
    minmin=X.min()
    maxmax=X.max()
    
    for i in range(test_num):
        for j in range(list_len):
            X[i][j]=(minmin+0.5*maxmax)/(X[i][j]+0.5*maxmax)
    ship_value=[]
    for i in range(test_num):
        ship_value.append(numpy.mean(X[i]))
    
    return ship_value
    





VC_FI.reverse()


X=[]
X.append(VC_FI)
X.append(VCE)


print(GCA_analysis(rul_list_down,X))












#%%
#
#x=list(range(len(VCE)))
#
#########################################画一条在197个点内，从1下降到0的曲线
#test_rul=[]
#for i in range(len(VCE)+3):
#    test_rul.append(i/(len(VCE)+1))
#test_rul_up=deepcopy(test_rul)
#test_rul.reverse()
#
##%%
#test_rul=feature_preprocessing(test_rul)
#test_rul_up=feature_preprocessing2(test_rul_up)
#
#
#TminV=Nondimensionalization(TminV)
#Cap=Nondimensionalization(Cap)
#VCE=Nondimensionalization(VCE)
#VC_FI=Nondimensionalization(VC_FI)
#test_rul=Nondimensionalization(test_rul)
#test_rul_up=Nondimensionalization(test_rul_up)
#plt.plot(x,test_rul, 'g-o',label='test_rul')
#plt.plot(x,test_rul_up, 'g-o',label='test_rul_up')
#########################################
#
#
#
#
#
#
#
#plt.plot(x,Cap,'m-^',label='Cap')      #6500           /1000      = 6.5---------------------------------
#plt.plot(x,VC_FI, 'g-o',label='VC_FI')     #40             /10            =4--------------------------------
#
#plt.legend(loc='upper right')
#plt.show()
#





#if __name__ == '__main__': main(r'G:\pythonProject\discharge_db\B0036.dat')



