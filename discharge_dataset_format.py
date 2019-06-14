# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:42:01 2018

@author: ZhengZhiyong

@Target:处理数据集，生成易于处理的格式

@Parameter:
    
    dataFile = r'G:\pythonProject\BatteryAgingARC-FY08Q4\B0005.mat'
    dataBaseFile = r'G:\pythonProject\discharge_db\B0005.dat'
    Battery = r'B0005'

"""



import scipy.io as scio #打开mat文件
from copy import deepcopy
import shelve #小型数据库



#打开源文件和目标数据库文件
def main(dataFile,dataBaseFile,Battery):
    data = scio.loadmat(dataFile)
    db = shelve.open(dataBaseFile, flag='c', protocol=None, writeback=False)
    
    db['discharge']=[]  #字典中放上一个列表
    temp_db=[]          #目标数据库文件缓存
    num = 0             #discharge个数
     
    for i in range(len(data[Battery][0][0][0][0])):             #遍历所有循环 600多个
    
        if data[Battery][0][0][0][0][i][0][0]=='discharge':  #对每一个discharge循环进行处理
    
            temp_dict={}    #临时字典
            temp_dict['Cycle']=num+1
            temp_dict['Am_temp']=data[Battery][0][0][0][0][i][1][0][0]
            temp_dict['Real_time']=deepcopy(list(data[Battery][0][0][0][0][i][2][0]))
            temp_dict['Voltage_measured']=deepcopy(list(data[Battery][0][0][0][0][i][3][0][0][0][0]))
            temp_dict['Current_measured']=deepcopy(list(data[Battery][0][0][0][0][i][3][0][0][1][0]))
            temp_dict['Temperature_measured']=deepcopy(list(data[Battery][0][0][0][0][i][3][0][0][2][0]))
            temp_dict['Current_load']=deepcopy(list(data[Battery][0][0][0][0][i][3][0][0][3][0]))
            temp_dict['Voltage_load']=deepcopy(list(data[Battery][0][0][0][0][i][3][0][0][4][0]))
            temp_dict['Time']=deepcopy(list(data[Battery][0][0][0][0][i][3][0][0][5][0]))
            temp_dict['Capacity']=data[Battery][0][0][0][0][i][3][0][0][6][0][0]
            
            temp_db.append(temp_dict)
            
            print(num) #打印次数
            num=num+1
            
    db['discharge']=temp_db       
                    
    db.close()
    
if __name__ == '__main__': main(r'G:\pythonProject\BatteryAgingARC_25-44\B0036.mat',r'G:\pythonProject\discharge_db\B0036.dat',r'B0036')



