
import csv
import linecache
import datetime
import sys
import numpy as np
import random
from scipy.sparse import *
from scipy import *
import scipy as sp
import matplotlib.pyplot as plt
import os.path
import matplotlib
import copy
import pandas as pd
from scipy import stats
from tempfile import TemporaryFile
import pickle
from scipy.integrate import odeint
from torchdiffeq import odeint as dodeint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

cudaQ = True
batch_size = 100

#if cudaQ:
#    print('???!!!')
#print('???')

def flushPrint(variable):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % variable)
    sys.stdout.flush()
#zhfont1 = matplotlib.font_manager.FontProperties(fname='/Users/zhangjiang/Library/Fonts/SimHei.ttf', size=16)

df1=pd.read_csv('county_city_province.csv')
cities1 = set((df1['CITY']))
df2=pd.read_csv('citypopulation.csv')
cities2 =set((df2['city']))
cities = set(list(cities1) + list(cities2))
nodes = {}
city_properties = {}
id_city = {}
for ct in cities:
    nodes[ct] = len(nodes)
    city_properties[ct] = {'pop':1,'prov':'','id':-1}
for i in df2.iterrows():
    city_properties[i[1][0]] = {'pop':float(i[1][1])}
for i in df1.iterrows():
    dc = city_properties.get(i[1]['CITY'],{})
    dc['prov'] = i[1]['PROV']
    dc['id'] = i[1]['CITY_ID']
    city_properties[i[1]['CITY']]=dc
    id_city[dc['id']] = i[1]['CITY']

    
df=pd.read_csv('city_flow_v1.csv')
flows = {}
for n,i in enumerate(df.iterrows()):
    if n % 1000 == 0:
        flushPrint(n/len(df))
    cityi=long(i[1]['cityi.id'])
    cityj=long(i[1]['cityj.id'])
    value = flows.get((cityi,cityj),0)
    flows[(cityi,cityj)] = value + i[1]['flowij']
    if cityi==341301:
        print(flows[(cityi,cityj)])
    
#存储到流量矩阵matrix
matrix = np.zeros([len(nodes), len(nodes)])
self_flux = np.zeros(len(nodes))
pij1 = np.zeros([len(nodes), len(nodes)])
for key,value in flows.items():
    id1 = nodes.get(id_city[key[0]],-1)
    id2 = nodes.get(id_city[key[1]],-1)
    matrix[id1, id2] = value
for i in range(matrix.shape[0]):
    self_flux[i] = matrix[i, i]
    matrix[i, i] = 0
    if np.sum(matrix[i,:])>0:
        pij1[i,:]=matrix[i,:]/np.sum(matrix[i,:])

df=pd.read_csv('Pij_BAIDU.csv',encoding='gbk')
df.head(4)

cities = {d:i for i,d in enumerate(df['Cities'])}
pij2 = np.zeros([len(nodes), len(nodes)])
for k,ind in cities.items():
    row = df[k]
    for city,column in cities.items():
        i_indx = nodes.get(city, -1)
        if i_indx < 0:
            print(city)
        j_indx = nodes.get(k, -1)
        if j_indx < 0:
            print(k)
        if i_indx >=0 and j_indx >= 0:
            pij2[j_indx, i_indx] = row[column] / 100
            if i_indx == j_indx:
                pij2[i_indx, j_indx] = 0
#pij = pij2

bools = pij2 <= 0
pij = np.zeros([pij1.shape[0], pij1.shape[0]])
for i in range(pij1.shape[0]):
    row = pij1[i]
    bool1 = bools[i]
    values = row * bool1
    if np.sum(values) > 0:
        ratios = values / np.sum(values)
        sum2 = np.sum(pij2[i, :])
        pij[i,:] = (1 - sum2) * ratios + pij2[i, :]
zeros = np.argwhere(np.sum(pij, axis=1) == 0).reshape(-1)
for idx in zeros:
    pij[idx][idx] = 1

np.sum(pij,1) #验证一下是否归一化

df = pd.read_csv('R_cases_cum.csv',encoding='gbk')
df

all_cases_cities = list(set(df['city_name']))[1:]

wuhan=df.loc[df['city_name']=='武汉市',['confirm','time','heal','dead']]
dates = list(wuhan['time'])
sorted_dates = np.sort(dates)
first_date = datetime.datetime(2020, 1, 1, 0, 0)
first_cases = int(wuhan.loc[wuhan['time']=='2020-01-01']['confirm'])


all_cases = {}

for city in all_cases_cities:
    subset = df.loc[df['city_name']==city,['confirm','time','heal','dead']]
    zhixia = False
    
    new_cases = np.array(subset['confirm'])
    cued_cases = np.array(subset['heal'])
    die_cases = np.array(subset['dead'])
    dates = list(subset['time'])
    days = []
    for i,dd in enumerate(dates):
        if pd.isnull(dd):
            dd = dates1[i]
        if not pd.isnull(dd):
            days.append((datetime.datetime.strptime(dd,'%Y-%m-%d') - first_date).days)
    sorted_days = np.sort(days)
    indx = np.argsort(days)
    infected = new_cases[indx]
    cued = cued_cases[indx]
    death = die_cases[indx]
    bools =sorted_days>=0
    if len(sorted_days)>0:
        all_cases[city] = (sorted_days[bools], infected[bools], cued[bools], death[bools])


# 不用可微分求解
def protect_decay(t, t0, eta, rate_time, relax):
    epsilon = 0.001
    r = 2 * np.log((1-epsilon)/epsilon) / rate_time
    x0 = t0 + rate_time/2
    decay = eta / (1 + np.exp(r * (t - x0))) + 1 - eta
    if relax>0:
        #tstar = x0 + np.log((1-relax*epsilon)/(relax*epsilon+eta-1))
        tstar = relax
        decay1 = eta/(1 + np.exp(r*(2*tstar - t - x0))) + 1 - eta
        decay = decay + decay1
    return decay
def diff(sicol, t, r_0, T_R, T_I, alpha, omega, eta, rate_time, protect_day, pijt, intervention, relax=0):
    sz = sicol.shape[0] // 3
    Inf = sicol[:sz]
    Conf = sicol[sz:2*sz]
    Sus = sicol[2*sz:]
    I_term = pijt.dot(Inf) - Inf
    C_term = pijt.dot(Conf) - Conf
    Sus_term = pijt.dot(Sus) - Sus
    cross_term = r_0  * Inf * Sus / T_I
    if intervention:
        cross_term = cross_term * protect_decay(t, protect_day, eta, rate_time, relax)
        
    delta_inf = cross_term - Inf / T_I + omega * I_term
    delta_conf = alpha * Inf / T_I - Conf / T_R #+ omega * C_term
    delta_sus = - cross_term + omega * Sus_term
    output = np.r_[delta_inf, delta_conf, delta_sus]
    return output
# 特定绘图函数
def plots(prediction,cities,plot_time_span,colors,colors_s):
    plt.figure(figsize=(15,10))

    for n,(k,i) in enumerate(nodes.items()):
        cityname = k
        ploty = prediction[:plot_time_span, i + len(nodes)]
        plt.semilogy(timespan[:plot_time_span], ploty * city_properties[k]['pop']
                     ,'-',alpha=0.1,color=colors[n])
        cityname1 = cityname
        if cityname[-1]=='市':
            cityname1 = cityname[:-1]
        itm = all_cases.get(cityname1,[])
        if len(itm)==0:
            itm = all_cases.get(cityname,[])
        if len(itm)>0:
            real = itm[1]-itm[2]-itm[3]
            #print(real)
            #plt.semilogy(itm[0], real, '.',color=colors[n],alpha=0.1)
            if k in cities:
                #print(itm[0])
                #print(real)
                c = cities.index(k)
                ploty =  prediction[:plot_time_span, i + len(nodes)]
                
                plt.semilogy(timespan[:plot_time_span], ploty * city_properties[k]['pop']
                             ,'-',color=colors_s[c])
                plt.semilogy(itm[0], real, 'o',color=colors_s[c])
    plt.show()
    
#可微分求微分方程，优化参数
#在无intervention时，优化参数有三个，如果考虑intervention，则t_m也可以被优化

class my_parameters(nn.Module):
    
    def __init__(self, batch_size, pij, T_I, T_R, omega, t_0=22, t_m=30, eta=1, intervention=False):
        super(my_parameters, self).__init__()
        self.batch_size = 1
        self.pij = pij
        self.T_I = T_I
        self.T_R = T_R
        self.omega = omega
        self.t0 = t_0
        self.tm = t_m
        self.eta = eta
        self.intervention = intervention
        
        if cudaQ:
            self.pps = Parameter(torch.rand([batch_size, 4]).cuda())
        else:
            self.pps = Parameter(torch.rand([batch_size, 4]))

        self.pps.data[:, 1] *= 500
        self.pps.data[:, 3] *= 100
    def form_parameter(self):
        if cudaQ:
            r0 = 4 * torch.sigmoid(self.pps) * torch.Tensor([[1,0,0,0]]).repeat(batch_size, 1).cuda()
            l0 = torch.relu(self.pps) * torch.Tensor([[0,1,0,0]]).repeat(batch_size, 1).cuda()
            alpha = torch.sigmoid(self.pps) * torch.Tensor([[0,0,1,0]]).repeat(batch_size, 1).cuda()
            tm = torch.relu(self.pps) * torch.Tensor([[0,0,0,1]]).repeat(batch_size, 1).cuda()
        else:
            r0 = 4 * torch.sigmoid(self.pps) * torch.Tensor([[1,0,0,0]]).repeat(batch_size, 1)
            l0 = torch.relu(self.pps) * torch.Tensor([[0,1,0,0]]).repeat(batch_size, 1)
            alpha = torch.sigmoid(self.pps) * torch.Tensor([[0,0,1,0]]).repeat(batch_size, 1)
            tm = torch.relu(self.pps) * torch.Tensor([[0,0,0,1]]).repeat(batch_size, 1)
        self.parameters = r0 + l0 + alpha + tm
        return r0+l0+alpha+tm
    def protect_decay(self, t, t0, eta, rate_time):
        epsilon = 0.001
        r = 2 * np.log((1-epsilon)/epsilon) / rate_time
        x0 = t0 + rate_time/2
        decay = eta / (1 + torch.exp(r * (t.unsqueeze(0).repeat(self.batch_size,1) - x0))) + 1 - eta
        return decay
    def forward(self, t, y):
        sz = y.size()[1] // 3
        Inf = y[:,:sz]
        Conf = y[:,sz:2*sz]
        Sus = y[:,2*sz:]
        I_term = Inf @ self.pij.t() - Inf
        C_term = Conf @ self.pij.t() - Conf
        Sus_term = Sus @ self.pij.t() - Sus
        parameter = self.parameters.unsqueeze(1).repeat(1,sz,1)
        cross_term = parameter[:, :, 0] * Inf * Sus / self.T_I
        if self.intervention:
            cross_term *= self.protect_decay(t, self.t0, self.eta, parameter[:, :, 3])
        delta_inf = cross_term - Inf / self.T_I + self.omega * I_term
        delta_conf = parameter[:, :, 2] * Inf /self.T_I - Conf / self.T_R #+ self.omega * C_term
        delta_sus = - cross_term + self.omega * Sus_term
        output = torch.cat((delta_inf, delta_conf, delta_sus),1)
        return output
#重新整理训练数据，加mask等
timespan = np.linspace(0, 200, 1000)
interval = 1000 / 200
infected = torch.zeros([len(timespan), len(nodes)])
recovered = torch.zeros([len(timespan), len(nodes)])
if cudaQ:
    mask = torch.zeros([len(timespan), len(nodes)]).cuda()
else:
    mask = torch.zeros([len(timespan), len(nodes)])
for city,itm in all_cases.items():
    city1 = city
    try:
        if city[-1] != '市':
            city1 = city + 'abs市'
        idx = nodes.get(city1, -1)
        if idx > 0:
            infected[(itm[0]*interval).astype(int), idx] = torch.Tensor(
                itm[1] - itm[2] - itm[3])
            bools = infected[:, idx] >0
            mask[bools, idx] = 1
            infected[(itm[0]*interval).astype(int), idx] = infected[(itm[0]*interval).astype(int), idx] / city_properties[city]['pop']
            recovered[(itm[0]*interval).astype(int),idx] = torch.Tensor((itm[2] + itm[3]
                                                      ) / city_properties[city]['pop'])
    except:
        continue
if cudaQ:
    targets = infected.cuda()
else:
    targets = infected
targets.size()

def DeExperiments(expn, savename, tr=False, ti=False, w=False, fc=False, trainn=50):
    # Experiments
    # T_I, T_R has uncertainty
    # T_R --> T_R
    # T_I --> T_I
   #batch_size = 100
    experiments = []
    free_params = []
    print(f'tr={tr}, ti={ti}, w={w}')

    for experiment in range(expn):
        #sample产生T_R,T_I
        if tr==False:
            T_R = np.random.randn()*0.9/1.96+9.2#9.2#10#1/0.0134#6 #serial interval，参考SARS
        else:
            T_R = tr
        if ti==False:
            T_I = np.random.randn()*0.4/1.96+8.3#8.3#8.89 #一个病患潜伏期时间，参考SARS、MERS
        else:
            T_I=ti
        if w==False:
            omega = np.random.randn()*0.008+0.03
        else:
            omega = w
        print(f'T_R={T_R}, T_I={T_I}, w={omega}')
        
        free_params = [T_R, T_I, omega]
        if cudaQ:
            nnn = my_parameters(batch_size, t_pij, T_I, T_R, omega).cuda()
        else:
            nnn = my_parameters(batch_size, t_pij, T_I, T_R, omega)
        optimizer = optim.Adam(nnn.parameters(), lr = 1)
        if cudaQ:
            # 感染，确诊，易感
            inf0 = torch.zeros([batch_size, len(nodes)]).cuda()
            conf0 = torch.zeros([batch_size, len(nodes)]).cuda()
            sus0 = torch.ones([batch_size, len(nodes)]).cuda()
        else:
            inf0 = torch.zeros([batch_size, len(nodes)])
            conf0 = torch.zeros([batch_size, len(nodes)])
            sus0 = torch.ones([batch_size, len(nodes)])
        inf0[:, nodes['武汉市']] = float(1)/float(city_properties['武汉市']['pop'])#1e-4 感染
        if fc==False:
            conf0[:, nodes['武汉市']] = float(first_cases)/float(city_properties['武汉市']['pop']) #确诊
        else:
            conf0[:, nodes['武汉市']] = float(fc)/float(city_properties['武汉市']['pop']) #确诊
        best_parameters = []

        for epoch in range(trainn):
            optimizer.zero_grad()
            pps = nnn.form_parameter()
            ls00 = inf0 * pps[:, 1].unsqueeze(1).repeat(1, len(nodes))
            sus0 = torch.ones([batch_size, len(nodes)])
            if cudaQ:
                sus0 = sus0.cuda()
            sus0[:,nodes['武汉市']]=1-ls00[:,nodes['武汉市']]-conf0[:,nodes['武汉市']]
            if cudaQ:
                prediction = dodeint(nnn, torch.cat((ls00, conf0, sus0), 1), torch.Tensor(timespan).cuda(),
                                    method='dopri5', rtol=1e-10)
            else:
                prediction = dodeint(nnn, torch.cat((ls00, conf0, sus0), 1), torch.Tensor(timespan),
                                    method='dopri5', rtol=1e-10)
            part1 = prediction[:, : len(nodes)]
            part2 = prediction[:, :, len(nodes): 2*len(nodes)]
            part3 = prediction[:, :, 2 * len(nodes) : ]
            whole = part2
            pred = torch.relu(whole * mask.unsqueeze(1).repeat(1, batch_size, 1))
            targ = targets * mask
            pred1 = (1 - torch.sign(pred)) + torch.sign(pred) * pred
            targ = (1 - torch.sign(targ)) + torch.sign(targ) * targ
            err = torch.log10(pred1) - torch.log10(targ.unsqueeze(1).repeat(1, batch_size, 1))
            all_loss = torch.mean(torch.mean(err ** 2, 0), 1)
            temp_loss = torch.mean(torch.mean(err**2, 0),0)
            loss = torch.mean(all_loss)
            print(experiment, epoch, loss)
            curr_min_loss, mindx = torch.min(all_loss, 0)

            #每个周期都记录最好的10个个体
            if len(best_parameters)<10:
                pps = nnn.form_parameter()
                if cudaQ:
                    output = pps[mindx:(mindx+1),:].cpu().data.numpy()
                else:
                    output = pps[mindx:(mindx+1),:].data.numpy()
                best_parameters.append([output, curr_min_loss.item(), free_params])
            else:
                all_fitness = sorted([[vvv[1],i] for i,vvv in enumerate(best_parameters)])
                if curr_min_loss < all_fitness[-1][0]:
                    #print(curr_min_loss)
                    del best_parameters[all_fitness[-1][1]]
                    pps = nnn.form_parameter()
                    if cudaQ:
                        output = pps[mindx:(mindx+1),:].cpu().data.numpy()
                    else:
                        output = pps[mindx:(mindx+1),:].data.numpy()
                    best_parameters.append([output, curr_min_loss.item(), free_params])
            loss.backward()
            optimizer.step()
            if cudaQ:
                nnn.pps.data = torch.where(torch.isnan(nnn.pps), torch.randn(nnn.pps.size()).cuda(), nnn.pps)
            else:
                nnn.pps.data = torch.where(torch.isnan(nnn.pps), torch.randn(nnn.pps.size()), nnn.pps)
        # save for each experiment
        experiments.append(best_parameters)
        f=open(savename,'wb')
        pickle.dump(experiments, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    # final saving
    f=open(savename,'wb')
    pickle.dump(experiments, f, pickle.HIGHEST_PROTOCOL)
    f.close()



#根据报告确定三个参数T_R,T_I,omega，优化搜索r0、alpha和latent的初始数字

#r0=2.2248602#2.6039007  
#initial_latent=304.56207
#alpha=0.9611629
#T_R = 9.2#10#1/0.0134#6 #serial interval，参考SARS #T_R = 9.2, var = [5, 9.2, 15]
#T_I = 8.3#8.89 #一个病患潜伏期时间，参考SARS、MERS # T_I, T_L = 8.3, var = [5, 8.3, 15]
#omega = 0.03#0.03#flowing_ratio # omega var = [0.01, 0.03, 0.05]
eta = 1
t_m = 35
t_0 = 22

if cudaQ:
    t_pij = torch.Tensor(pij).cuda()
else:
    t_pij = torch.Tensor(pij)
#batch_size = 100

import time

begin = time.time()

num_of_fits=3
training_epco=50

DeExperiments(num_of_fits, 'experiments_ti_tr_3_test.pkl', trainn=training_epco)
#DeExperiments(num_of_fits, 'sensitivity_tr_15.pkl', tr=15.0, trainn=nn)
#DeExperiments(num_of_fits, 'sensitivity_ti_5.pkl', ti=5.0, trainn=nn)
#DeExperiments(num_of_fits, 'sensitivity_ti_15.pkl', ti=15.0, trainn=nn)
#DeExperiments(num_of_fits, 'sensitivity_w_0.01.pkl', w=0.01, trainn=nn)
#DeExperiments(num_of_fits, 'sensitivity_w_0.05.pkl', w=0.05, trainn=nn)

#for ti in [5, 15]:
#    print('T_R=',ti,'...')
#    DoExpeiment(120, 'var_ti_experiments'+str(ti)+'.pkl', T_R=ti)
#    print('time:', time.time() - begin, 's')
    
#for tl in [5, 15]:
#    print('T_I=',tl,'...')
#    DoExpeiment(120, 'var_tl_experiments'+str(tl)+'.pkl', T_I=tl)
#    print('time:', time.time() - begin, 's')
    
#for ga in [0.01, 0.05]:
#    print('omega=',ga,'...')
#    DoExpeiment(120, 'var_gamma_experiments'+str(ga)+'.pkl', omega=ga)
#    print('time:', time.time() - begin, 's')
    
end = time.time()

print('finished, time useage:', end-begin, 's')
