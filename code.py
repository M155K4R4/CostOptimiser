
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:49:03 2016

@author: Alienware
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:36:37 2016

@author: shubhankar.mitra
"""
import os
os.system('C:\WinPython-64bit-3.4.4.1\python-3.4.4.amd64\Scripts\pip.exe install pandasql')
import pandas as pd
import numpy as np
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
import math
from scipy import optimize
import matplotlib.pyplot as plt

dt_IHR=pd.read_excel("D:/Spyder/mmm/InHome-Repair_Data_20131415_ver1.2_2.xlsx")

q="""select DMA_reg_comb,sub_channel,Division,
count(cost) 
,sum(case when cost<>0 then 1 else 0 end) as non_zero_cost
,sum(case when Created_orders<>0 then 1 else 0 end) as non_zero_orders
,sum(case when Created_orders>0 and cost>0 then 1 else 0 end) as non_zero_CostAndOrd
from dt_IHR
group by DMA_reg_comb,sub_channel,Division
"""
explr1=pysqldf(q)
##Inhome repair



q3="""
select a.acctg_yr,a.acctg_mth,a.DMA_reg_comb,a.Division,a.sub_channel,a.cost,a.Created_orders from
dt_IHR a
inner join
(select DMA_reg_comb,sub_channel,Division,case when non_zero_CostAndOrd>=30 then 1 else 0 end as sub_channel_Wdata
from explr1) b
on a.DMA_reg_comb=b.DMA_reg_comb  
and a.sub_channel=b.sub_channel
and a.Division=b.Division
and a.cost>0
and a.Created_orders>0
and b.sub_channel_Wdata=1
"""

dataW30Rows=pysqldf(q3)


dataW30Rows.cost=dataW30Rows.cost.round(2)
dataW30Rows.Created_orders=np.int64(dataW30Rows.Created_orders)

dt_Chcg_ckdsh=dataW30Rows.loc[(dataW30Rows.DMA_reg_comb=='Chicago IL')&(dataW30Rows.Division=='cook/dish')]



def plotter(x,y,channel='fig'):
    fig, axes = plt.subplots()
    fig.canvas.set_window_title(channel)
    axes.scatter(x,y)


#dt_Chcg_ckdsh.groupby(['sub_channel']).apply(lambda x: plotter(x.cost,x.Created_orders,x.sub_channel.unique()[0]))

dt_Chcg_ckdsh_WP=dt_Chcg_ckdsh.loc[(dt_Chcg_ckdsh.sub_channel=='Display')]


def fun1(c,x,y=0):
     return c[0]*np.exp(-c[1]*(np.exp(-c[2]*(x))))-y

def rem_outlier(ds,col):
    #print(len(ds))
    #print((ds[col].mean()-ds[col].median())/ds[col].median())
    if  (ds[col].mean()-ds[col].median())/ds[col].median()>.2:
        #print(ds[ds[col]==ds[col].max()])
        ds=ds[ds[col]!=ds[col].max()]
        return rem_outlier(ds,col)
    elif (ds[col].mean()-ds[col].median())/ds[col].median()<-.2:
        #print(ds[ds[col]==ds[col].min()])
        ds=ds[ds[col]!=ds[col].min()]
        return rem_outlier(ds,col)
    else:
        return(ds)

class func:
    def __init__(self,const):        
        self.c=const
        
    def fun2(self,x):
        return self.c[0]*np.exp(-self.c[1]*(np.exp(-self.c[2]*(x))))
        


def solver(fn,start,end,target,cost_norm,order_norm):
    iter=start
    while iter<end and (fn(iter/cost_norm)*order_norm)-target<0:
        iter=iter+1
    return iter





def createModel(ds):
    print(ds.Division.unique()[0]+ds.sub_channel.unique()[0])
    
    ds_old=ds
    ds=rem_outlier(ds,'cost')
    ds=rem_outlier(ds,'Created_orders')
    t_train=ds.cost/ds.cost.max()
    y_train=ds.Created_orders/ds.Created_orders.max()
    res_robust = optimize.least_squares(fun1, [1,1,1], loss='huber', f_scale=0.1, args=(t_train, y_train))
    #res_lssq = optimize.least_squares(fun1, [1,1,1], f_scale=0.1, args=(t_train, y_train))
    c=res_robust.x
    
    Max_Ord=.9*fun1(c,1,0)*ds.Created_orders.max()
    Min_Ord=1.1*fun1(c,ds.cost.min()/ds.cost.max(),0)*ds.Created_orders.max()
    
    MaxCostFinder=solver(func(c).fun2,ds.cost.min(),ds.cost.max(),Max_Ord,ds.cost.max(),ds.Created_orders.max())
    Upper_Cost_Limit=MaxCostFinder
    MinCostFinder=solver(func(c).fun2,ds.cost.min(),ds.cost.max(),Min_Ord,ds.cost.max(),ds.Created_orders.max())
    Lower_Cost_Limit=MinCostFinder
    #d=res_lssq.x
    fig, axes = plt.subplots()
    fig.canvas.set_window_title(ds.Division.unique()[0]+ds.sub_channel.unique()[0])
    axes.scatter(ds_old.cost,ds_old.Created_orders,color='r')
    axes.scatter(ds.cost,ds.Created_orders,color='g')
    axes.scatter(Lower_Cost_Limit,func(c).fun2(Lower_Cost_Limit/ds.cost.max())*ds.Created_orders.max(),color='y')
    axes.scatter(Upper_Cost_Limit,func(c).fun2(Upper_Cost_Limit/ds.cost.max())*ds.Created_orders.max(),color='y')
    axes.plot(range(0,int(ds.cost.max()),1),[fun1(c,x/ds.cost.max(),0)*ds.Created_orders.max() for x in range(0,int(ds.cost.max()),1)],color='b')
    #axes.plot(range(3,int(ds.cost.max()),1),np.diff(np.diff(np.diff(np.array([fun1(c,x/ds.cost.max(),0)*ds.Created_orders.max() for x in range(0,int(ds.cost.max()),1)])))))
    print('done')    
    return [c,Upper_Cost_Limit,Lower_Cost_Limit,ds.cost.max(),ds.Created_orders.max()]
    #axes.plot(range(0,int(ds.cost.max()),1),[fun1(d,x/ds.cost.max(),0)*ds.Created_orders.max() for x in range(0,int(ds.cost.max()),1)],color='g')
    
groups=dt_Chcg_ckdsh.groupby(['sub_channel'])
res=groups.apply(lambda x: createModel(x))



class optClass():
    def __init__(self,modelParams):
        self.res=modelParams
    
    def optFunc(self,inp):
        sub_Channels=self.res.groupby(self.res.index)
        res2=sub_Channels.apply(lambda x:func(x[0][0]).fun2(inp[np.where(self.res.index==x.index[0])[0][0]]/x[0][3])*x[0][4])
        return -sum(res2)
  

class constraints:
    def __init__(self,maxValue):
        self.maxValue=maxValue
    def cons1(self,x):
        return sum(x)-self.maxValue

cons=[{'type':'eq', 'fun': constraints(1563).cons1}]

res2=optimize.minimize(optClass(res).optFunc,[(x[2].round()+x[1].round())/2 for x in res], method='TNC'
,bounds=[(x[2].round(),x[1].round()) for x in res], constraints=cons)
   
optClass(res).optFunc([(x[2].round()+x[1].round())/2 for x in res])

optClass(res).optFunc([x[2].round() for x in res])
optClass(res).optFunc([x[1].round() for x in res])

optClass(res).optFunc(res2.x)

