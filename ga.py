import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_solve import data_deal
import random 
# from fjsp import FJSP
# import matplotlib.pyplot as plt 
#plt.rcParams['font.sans-serif'] = ['STSong'] 
# from matplotlib.pylab import mpl
import itertools
from collections import Counter
import copy
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


class GA():

    def __init__(self,generation,popsize,p1,p2,parm_data,machine_num):
        self.generation=generation                  #迭代次数
        self.popsize = popsize                      # 种群规模
        self.p1=p1                                  #交叉的概率
        self.p2=p2                                  #突变的概率
        # self.to=to                                  #环境
        self.Tmachine,self.Tmachinetime,self.tdx,self.work,self.tom,self.machines=parm_data[0],parm_data[1],parm_data[2],parm_data[3],parm_data[4],parm_data[5]
       
        # self.w,self.m,self.t=[],[],[] #已经生成的方案        
        self.machine_num = machine_num
        self.job_num=len(self.machines)
        self.operation_num =sum(self.machines)
        self.job_list = [i for i in range(self.job_num)]
        self.GS_num=int(0.1*self.popsize)      #全局选择初始化
        self.LS_num=int(0.5*self.popsize)     #局部选择初始化
        self.RS_num=int(0.4*self.popsize)     #随机选择初始化

    def find_index(self,list_M,list_S,list_W,time):#list_M(可无),list_S,list_W原调度的机器，开工时间、工序加工时间的安排(对应job_operation列表)
        car=[]#被卡住的序号和工作剩余时间
        record = []
        for i in range(len(list_M)):
            if(list_S[i]<time):
                record.append(i)
                if (list_S[i]+list_W[i]>time):
                    car.append((i,list_S[i]+list_W[i]-time))
        if not car:
            return [(max(record),0)],record
        else:
            return car, record


    def CHS_Matrix(self, C_num):#C_num指种群个数
        return np.zeros([C_num, len(self.work)], dtype=int)

    def Global_initial(self):
        MS = self.CHS_Matrix(self.GS_num)
        OS_list = copy.deepcopy(self.work)
        OS = self.CHS_Matrix(self.GS_num)
        for i in range(self.GS_num): 
            # Machine_time = np.zeros(self.machine_num, dtype=int)  # 机器时间初始化
            random.shuffle(OS_list)  # 生成工序排序部分
            OS[i] = np.array(OS_list)
            GJ_list = copy.deepcopy(self.job_list)

            random.shuffle(GJ_list)
            list_load = [0]*self.machine_num#
            for g in GJ_list:  # 随机选择工件集的第一个工件,从工件集中剔除这个工件 #对应工件编号
                for j in range(self.machines[g]):# 从工件的第一个工序开始选择机器
                    highs = self.tom[g][j]
                    if j == 0:
                        lows = 0
                    else:
                        lows = self.tom[g][j]-self.tdx[g][j]
 
                    machine_feasible =[i-1 for i in self.Tmachine[g][lows:highs]]#可选的机器index


                    corr_time = self.Tmachinetime[g][lows:highs]

                    index = np.argmin([list_load[i]+j for i,j in zip(machine_feasible,corr_time)])#对应的负荷处理时间最小的那台机器
                    index1 = machine_feasible[index]#对应的负荷处理时间最小的那台机器 
                    list_load[index1] += corr_time[index]
                    index2 = sum(self.machines[:g])+j 

                    MS[i][index2] = index

        CHS1 = np.hstack((MS, OS))

        return CHS1
    
    def Local_initial(self):
        MS = self.CHS_Matrix(self.LS_num)
        OS_list = copy.deepcopy(self.work)
        OS = self.CHS_Matrix(self.LS_num)
        for i in range(self.LS_num): 
            # Machine_time = np.zeros(self.machine_num, dtype=int)  # 机器时间初始化
            random.shuffle(OS_list)  # 生成工序排序部分
            OS[i] = np.array(OS_list)
            GJ_list = self.job_list
            list_load = [0]*self.machine_num#
            for g in GJ_list:  # 选择工件集的第一个工件,从工件集中剔除这个工件
                for j in range(self.machines[g]):# 从工件的第一个工序开始
                    highs = self.tom[g][j]
                    if j == 0:
                        lows = 0
                    else:
                        lows = self.tom[g][j]-self.tdx[g][j]
                # 					highs=self.tom[signal][count[signal,0]]
                    # print('fff',self.Tmachine[g][lows:highs])
                    machine_feasible = list(map(lambda x:x-1,self.Tmachine[g][lows:highs])) #可选的机器index
                    # print('fff',machine_feasible)                
                    # machine_feasible = list(map(lambda x:x-1,self.Tmachine[g][lows:highs]))  #可选的机器index
                    # print(machine_feasible)
                    corr_time = self.Tmachinetime[g][lows:highs]
                    index = np.argmin([list_load[i]+j for i,j in zip(machine_feasible,corr_time)])#对应的负荷处理时间
                    index1 = machine_feasible[index]
                    list_load[index1] += corr_time[index]
                    index2 = sum(self.machines[:g])+j
                    MS[i][index2] = index
        CHS1 = np.hstack((MS, OS))
        return CHS1
    
    def Random_initial(self):
        MS = self.CHS_Matrix(self.RS_num)
        OS_list = copy.deepcopy(self.work)
        OS = self.CHS_Matrix(self.RS_num)
        for i in range(self.RS_num): 
            # Machine_time = np.zeros(self.machine_num, dtype=int)  # 机器时间初始化
            random.shuffle(OS_list)  # 生成工序排序部分
            OS[i] = np.array(OS_list)
            GJ_list = copy.deepcopy(self.job_list)
            for g in GJ_list: 
                for j in range(self.machines[g]):
                    index = sum(self.machines[:g])+j
                    MS[i][index] = np.random.randint(self.tdx[g][j])
        CHS1 = np.hstack((MS, OS))
        return CHS1
    
    def decode(self,chrom):
        l = len(chrom)

        MS, OS = chrom[:int(l/2)], chrom[int(l/2):]
        count = [-1]*self.job_num
        # print(self.job_num)
        M_list = []
        T_list = []
        # print("MS={},l={}".format(MS,len(MS)))
        # print("OS={},l={}".format(OS,len(OS)))
        for i in OS:
            count[i]+=1
            index = sum(self.machines[:i])+count[i]
            low =  self.tom[i][count[i]]-self.tdx[i][count[i]]
            M_list.append(self.Tmachine[i][low+MS[index]])
            T_list.append(self.Tmachine[i][low+MS[index]])
        return np.array(OS), np.array(M_list), np.array(T_list)


        #机器部分交叉
    def Crossover_Machine(self,CHS1,CHS2):
        
        T_r=[j for j in range(self.operation_num)]
        r = random.randint(1, self.operation_num-1)  # 在区间[1,T0]内产生一个整数r
        random.shuffle(T_r)
        R = T_r[0:r]  # 按照随机数r产生r个互不相等的整数
        # 将父代的染色体复制到子代中去，保持他们的顺序和位置
        P1 = CHS1
        P2 = CHS2
        for i in R:
            K,K_2 = P1[i],P2[i]
            CHS1[i],CHS2[i] = K_2,K
        return CHS1,CHS2

    #工序交叉部分
    def Crossover_Operation(self,CHS1, CHS2):
        Job_list = copy.deepcopy(self.job_list)
        random.shuffle(Job_list)
        r = random.randint(1, len(Job_list) - 1)
        Set1 = Job_list[0:r]
        Set2 = Job_list[r:]
        # print('l',self.operation_num)
        
        new_CHS1 = np.zeros(self.operation_num, dtype=int)
        new_CHS2 = np.zeros(self.operation_num, dtype=int)
        part1,part2 = [[],[]],[[],[]]
        CHS1, CHS2 = CHS1.tolist(), CHS2.tolist()
        # print(type(Set1))
        # chs1,chs2 = CHS1,CHS2
        #被固定的染色体片段
        # print('CHS1',CHS1)
        for k1, v1 in enumerate(CHS1):
            # print('v1',v1)
            # print('Set1',Set1)
            if v1 in Set1:
                part2[0].append(k1)#position
                part2[1].append(v1)#value
            # if v1 in Set2:
            #     del_index1.append(k1)
        for k2, v2 in enumerate(CHS2):
            if v2 in Set2:
                part1[0].append(k2)
                part1[1].append(v2)
            # if v2 in Set1:
            #     del_index2.append(k2)
        chs1 = part2[1]
        chs2 = part1[1]
        # print(len(chs1))
        # print(len(part1[0]))
        # chs1 = [i for num,i in enumerate(CHS1) if num not in del_index1]
        # chs2 = [i for num,i in enumerate(CHS2) if num not in del_index2]
        # print('set1',Set1)
        # print('set2',Set2)
        # print('CHS1',CHS1)
        # print('part1',part1)
        # print('CHS2',CHS2)
        # print('part2',part2)
        # print('chs1',chs1)
        # print('chs2',chs2)
        #new2 = chs2 + part2(from chs1) 
        #new1 = chs1 + part1(from chs2)
        # print(len(chs1),len(chs2))
        count1, count2= 0, 0
        new_CHS1[part1[0]] = part1[1]
        new_CHS2[part2[0]] = part2[1]
        for i in range(self.operation_num):
            # print(count1,count2)
            if i not in part1[0]:
                new_CHS1[i] = chs1[count1]
                count1 += 1
                # print(new_CHS1)
            if i not in part2[0]:
                new_CHS2[i] = chs2[count2]
                count2 += 1
                # print(new_CHS1)
        return np.array(new_CHS1),np.array(new_CHS2)

    
    # def ma_mul(self,job,machine,machine_time): #mutation
    #     count=np.zeros((self.job_num,1),dtype=np.int)-1
    #     j=0
    #     r=np.random.randint(1,job.shape[1]-index)
    #     idx=np.random.randint(index+1,job.shape[1],r)
    #     idx=list(set(idx))
    #     for i in range(job.shape[1]):
    #         svg=int(job[0,i])
    #         count[svg,0]+=1
    #         if(i==idx[j]):
    #             if(j<len(idx)-1):
    #                 j+=1
    #             highs=self.tom[svg][count[svg,0]]
    #             lows=self.tom[svg][count[svg,0]]-self.tdx[svg][count[svg,0]]
    #             n_machine=self.Tmachine[svg,lows:highs].tolist()
    #             n_time=self.Tmachinetime[svg,lows:highs].tolist()
    #             loc_idx=n_time.index(min(n_time))
    #             machine[0,i]=n_machine[loc_idx]
    #             machine_time[0,i]=min(n_time)
    #     return machine,machine_time
    
    #机器变异部分
    def Mutation_Machine(self,job,machine,machine_time):
        Tr=[i_num for i_num in range(self.operation_num)]
        # 机器选择部分
        r = random.randint(1, self.operation_num - 1)  # 在变异染色体中选择r个位置
        random.shuffle(Tr)
        T_r = Tr[0:r]
        count = [-1]*self.job_num
        # print('tom',self.tom)
        # print('job',job)
        for i in T_r:
            Job=job[i]
            count[Job] += 1
            #找到对应工序可用的机器
            # print('Job={},count={}'.format(Job,count))
            highs=self.tom[Job][count[Job]]
            lows=self.tom[Job][count[Job]]-self.tdx[Job][count[Job]]                
            n_machine=self.Tmachine[Job][lows:highs]
            n_time=self.Tmachinetime[Job][lows:highs]
            loc_idx=n_time.index(min(n_time))
            machine[i]=n_machine[loc_idx]
            machine_time[i]=min(n_time)
        return machine,machine_time
    
    #工序变异部分
    def Mutation_Operation(self,job,machine,machine_time):
        
        # r=random.randint(1,self.job_num-1)
        r = 4
        result = []
        Tr = np.random.choice(self.operation_num,r,False)#被选中的工件(位置)
        select = {}
        Tr1 = []
        #去重
        for i in Tr : #i表示的是位置 
            if job[i] in select:#选中的位置有重复的工件编号
                select[job[i]].append(i)
            else:
                select.update({job[i]:[i]})
        if len(select) == 1:
            return job
        else:
            for key in select:
                Tr1.append(random.choice(select[key])) 
            #组合所有结果
            # print('select',select)
            # print('Tr1',Tr1)
            # print('job',job)
            # print('fff',job[Tr1])
            set = list (itertools.permutations(job[Tr1]))
            # print('job',job)
            # print(set)
            job1 = []
            for i in range(1, len(set)):
                job[Tr1] = set[i]
                job1.append(job)
                C_finish,_,_=self.caculate(job,machine,machine_time)
                result.append(C_finish)  #TODO 可能是空集 
            return job1[result.index(min(result))]
    
    def caculate(self,job,machine,machine_time):
        jobtime = [0]*self.job_num    #上一个工序的结束时间
        tmm=[0]*self.machine_num     	#tmm机器可使用的最早时间							
        startime=0
        list_S,list_W=[],[]
        for i in range(len(job)):
            svg=job[i]
            sig=machine[i]-1
            startime=max(jobtime[svg],tmm[sig])   	
            tmm[sig]=startime+machine_time[i]
            jobtime[svg]=startime+machine_time[i]
            # list_M.append(sig+1)
            list_S.append(startime)
            list_W.append(machine_time[i])
        C_finish=max(tmm)			#最晚完工时间
        return C_finish,list_S,list_W

    def caculate1(self,job,machine,machine_time,error_M_T):
        jobtime = [0]*self.job_num    #上一个工序的结束时间
        tmm=[0]*self.machine_num    	#tmm机器可使用的最早时间
        for j, time in error_M_T.items():
            tmm[j[0]-1] = time
            jobtime[j[1]] = time
        startime=0
        list_S,list_W=[],[]
        for i in range(len(job)):#遍历工序单（基因）
            svg=job[i]
            sig=machine[i]-1
            startime=max(jobtime[svg],tmm[sig])
            tmm[sig]=startime+machine_time[i]
            jobtime[svg]=startime+machine_time[i]
            list_S.append(startime)
            list_W.append(machine_time[i])
        C_finish=max(tmm)			#最晚完工时间
        return C_finish,list_S,list_W
    
    def ga_initial(self):
        answer=[] #种群中每个sol的make_span
        result=[] #最优的个体完成时间
        CHS = np.zeros((self.popsize,2*len(self.work)),dtype = int)
        work_job1,work_job=np.zeros((self.popsize,len(self.work)),dtype = int),np.zeros((self.popsize,len(self.work)),dtype = int)
        work_M1,work_M=np.zeros((self.popsize,len(self.work)),dtype = int),np.zeros((self.popsize,len(self.work)),dtype = int)
        work_T1,work_T=np.zeros((self.popsize,len(self.work)),dtype = int),np.zeros((self.popsize,len(self.work)),dtype = int)
        CHS[:self.GS_num] = self.Global_initial()
        CHS[self.GS_num:self.GS_num+self.LS_num] =  self.Local_initial()
        CHS[self.popsize-self.RS_num:] =  self.Random_initial()
        np.random.shuffle(CHS)        
        for i in range(self.popsize):
            chrom = CHS[i]
            work_job[i],work_M[i],work_T[i] = self.decode(chrom)
            C_finish,_,_ =self.caculate(work_job[i],work_M[i],work_T[i])
            answer.append(C_finish)
        for gen in range(self.generation):
            answer1=[]
            work_jobb,work_MM,work_TT=np.copy(work_job),np.copy(work_M),np.copy(work_T) #拷贝更新的结果
            work_job3,work_M3,work_T3=np.copy(work_job1),np.copy(work_M1),np.copy(work_T1)
            # print('gen=',gen)
            for i in range(0,self.popsize,2):#相邻两个亲代结合
                W1,M1,T1=work_jobb[i],work_MM[i],work_TT[i] #亲代1
                W2,M2,T2=work_jobb[i+1],work_MM[i+1],work_TT[i+1]#亲代2
                # print(W1)
                if np.random.rand()<self.p1:
                    W1,W2 = self.Crossover_Operation(W1,W2)
                    M1,M2 = self.Crossover_Machine(M1,M2)
                if np.random.rand()<self.p2:
                    M1,T1=self.Mutation_Machine(W1,M1,T1)
                    M2,T2=self.Mutation_Machine(W2,M2,T2)
                    W1 = self.Mutation_Operation(W1,M1,T1)
                    W2 = self.Mutation_Operation(W2,M2,T2)

                C_finish,_,_=self.caculate(W1,M1,T1)
                work_job3[i]=W1 #更新工序编码
                answer1.append(C_finish)
                work_M3[i]=M1
                work_T3[i]=T1

                C_finish,_,_=self.caculate(W2,M2,T2)
                work_job3[i+1]=W2  #更新工序编码
                answer1.append(C_finish)
                work_M3[i+1]=M2
                work_T3[i+1]=T2
            work_job2,work_M2,work_T2=np.vstack((work_job,work_job3)),np.vstack((work_M,work_M3)),np.vstack((work_T,work_T3))
            answer2=answer+answer1#亲代子代结合比较
            best_idx=np.array(answer2).argsort()[0:self.popsize] 
            work_job,work_M,work_T=work_job2[best_idx],work_M2[best_idx],work_T2[best_idx]
        
            answer=np.array(answer2)[best_idx].tolist()
            best_index=answer2.index(min(answer2))             #找到最小完工时间的个体
        
            result.append([gen+1,min(answer2)])#记录每一次迭代的最优个体
        return work_job2[best_index],work_M2[best_index],work_T2[best_index],result 
    
    def draw_Gantt(self,data_list,arrive_time,initial_time,end_range):
        columns = ['machine', 'start_time', 'end_time', 'job']
        df = pd.DataFrame(data_list, columns=columns)
        df = df.sort_values(by=['start_time'])
        base_datetime = pd.Timestamp(initial_time)
        df['job'] = df['job'].astype(str)
        df['start_time'] = base_datetime + pd.to_timedelta(df['start_time'], unit='H')
        df['end_time'] = base_datetime + pd.to_timedelta(df['end_time'], unit='H')
        # print(df)
        df.to_excel('example.xlsx', index=False)

        vertical_lines = copy.deepcopy(arrive_time)
        for i in range(len(arrive_time)):
            vertical_lines[i] = base_datetime + timedelta(hours=vertical_lines[i])
            vertical_lines[i] = vertical_lines[i].strftime('%Y-%m-%d %H:%M:%S')


        end_range = (base_datetime + timedelta(hours=end_range)).strftime('%Y-%m-%d %H:%M:%S')
        vertical_lines.append(end_range)
            

        # generate Gantt chart
        fig = px.timeline(
            df,
            x_start="start_time",
            x_end="end_time",
            y="machine",
            color="job",
            # color_discrete_map=dict(zip(resource_colors['job'], resource_colors['job'])),  # 设置颜色映射
            labels={"start_time": "start_time", "end_time": "end_time"},
            title="schedule"
        )
        for date in vertical_lines:
            fig.add_shape(
                type="line",
                x0=date,
                x1=date,
                y0=0,
                y1=8,
                line=dict(color="black", dash="dash")
            )
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row['start_time'] + (row['end_time'] - row['start_time']) / 2,
                y=row['machine'],
                text=row['job'],
                showarrow=False,
                font=dict(size=8)
            )
        
        fig.show()
        fig.write_html("gantt_chart1.html")
        # fig.write_image("gantt_chart1.jpg")



            
    ################有新车加入时怎么计算#######################
    def ga_total(self,new_car,initial_time="2023-09-01 08:00:00"):
        #new_car[[...]] arrive time[] len(new_car) = len(arrive_time)
        # task:machine type, start_time,finish_time,resource:job
        data_list=[]
        job, machine, machine_time, _ = self.ga_initial()#得到的初始结果
        _,list_S,list_W=self.caculate(job,machine,machine_time)
        arrive_time = list(new_car.keys())
        for k in range(len(arrive_time)):
            #更新tdx Tmachine Tmachinetime
            #machines[6, 5, 5, 5, 6, 6, 5, 5, 6, 6]每个工件包含的工序数 tom累加 
            if k == 0:
                time = arrive_time[k]
            else:
                time = arrive_time[k]-arrive_time[k-1]

            index,finish=self.find_index(machine,list_S,list_W,time)#[(工序在表单中的位置，剩余时间)]

            if k == 0:
                data_list += [(machine[i],list_S[i],list_S[i]+list_W[i],job[i]) for i in finish]
            else:
                data_list += [(machine[i],list_S[i]+arrive_time[k-1],list_S[i]+list_W[i]+arrive_time[k-1],job[i]) for i in finish]
    

            error_M_T = {}
            for i in range(len(index)):
                error_M_T.update({(machine[index[i][0]],job[index[i][0]]):index[i][1]}) #故障机器,正在进行中的工件:#剩余时间（故障时间）

             ######FJSP的充分完备信息更新

            finish = [job[i] for i in finish]
            counter1 = dict(Counter(job))
            counter1 = dict(sorted(counter1.items(), key=lambda v:v[0]))
            counter2 = dict(Counter(finish))
            counter2 = dict(sorted(counter2.items(), key=lambda v:v[0]))
            counter = dict(Counter(counter1)-Counter(counter2))
            self.job_list = list(counter.keys())
            work = []
            Tmachine = copy.deepcopy(self.Tmachine)
            machines  = copy.deepcopy(self.machines)
            Tmachinetime = copy.deepcopy(self.Tmachinetime)
            tdx = copy.deepcopy(self.tdx)
            tom = copy.deepcopy(self.tom)
            

            for key,value in counter.items(): #key:工件，value:剩余工序数，如果已经完成则不在考虑之中{1:4,3:5...}

                ind = self.machines[key]-value #已经完成的工件则不会更新，也不会用到

                work += [key]*value
                machines[key] = value
                tdx[key] = self.tdx[key][ind:]

                if ind!=0:
                    tom[key] = [i-self.tom[key][ind-1] for i in self.tom[key][ind:]]
                    Tmachine[key] = self.Tmachine[key][self.tom[key][ind-1]:]
                    Tmachinetime[key] = self.Tmachinetime[key][self.tom[key][ind-1]:]

            car = copy.deepcopy(new_car[arrive_time[k]])
            self.job_list.append(self.job_num)  
            num = [self.job_num]*car[0]
            work += num
            self.job_num += 1

            Tmachine.append(car[1])
            Tmachinetime.append(car[2])
            machines.append(car[0])
            tdx.append(car[3])
            tom.append(car[4])            
            ######FJSP的充分完备信息更新#########
            self.work = copy.deepcopy(work) #更新work
            self.machines =  copy.deepcopy(machines)#重新修剪，将已经完成安排的工序剪掉
            self.Tmachine = copy.deepcopy(Tmachine)#重新修剪，将已经完成安排的工序剪掉
            self.Tmachinetime = copy.deepcopy(Tmachinetime)#重新修剪，将已经完成安排的工序剪掉
            self.tdx = copy.deepcopy(tdx) #重新修剪，将已经完成安排的工序剪掉
            self.tom = copy.deepcopy(tom)
            self.operation_num = sum(np.array(self.machines)[self.job_list])
            # print('work={},machines={},Tmac={},Tmacintine={},tdx={},tom={}'.format(self.work,self.machines,self.Tmachine,self.Tmachinetime,self.tdx,self.tom))
            ####计算得到哪几个机器现在还不能用（机器故障）#####
    ##################### 修剪param_data &&&&& 考虑多个机器故障 ########################
    ###########更新Tmachine,Tmachinetime,tdx,work,tom,machines###################
                   ####计算得到哪几个机器现在还不能用（机器故障）#####
            #######error_S = 0; error_M+error_T(index可算出)################

            answer=[] #种群中每个sol的make_span
            result=[] #最优的个体完成时间
            #每次生成的结果包含工序单、机器单、处理时间单
            #work_job1初始化空值 work_job亲代  work_job3 子代
            work_job1,work_job=np.zeros((self.popsize,len(self.work)),dtype = int),np.zeros((self.popsize,len(self.work)),dtype = int)
            work_M1,work_M=np.zeros((self.popsize,len(self.work)),dtype = int),np.zeros((self.popsize,len(self.work)),dtype = int)
            work_T1,work_T=np.zeros((self.popsize,len(self.work))),np.zeros((self.popsize,len(self.work)))

            print('eeroe',error_M_T)
            for gen in range(self.generation):
                print('gen_new=',gen)
                if(gen<1):                      #第一次生成多个可行的工序编码，机器编码，时间编码
                    for i in range(self.popsize):
                        # print('num',i)
                        job, machine, machine_time, _ = self.ga_initial()

                        C_finish,list_S,_=self.caculate1(job,machine,machine_time,error_M_T) #出现故障后计算得到的完成时间
                        answer.append(C_finish)
                        work_job[i]=(job)
                        work_M[i]=machine
                        work_T[i]=machine_time
                    result.append([gen,min(answer)])#记录初始解的最小完工时间


                answer1=[]

                work_jobb,work_MM,work_TT=np.copy(work_job),np.copy(work_M),np.copy(work_T) #拷贝更新的结果
                work_job3,work_M3,work_T3=np.copy(work_job1),np.copy(work_M1),np.copy(work_T1)
                for i in range(0,self.popsize,2):#相邻两个亲代结合
                    print('individual=',i)
                    W1,M1,T1=work_jobb[i],work_MM[i],work_TT[i] #亲代1
                    W2,M2,T2=work_jobb[i+1],work_MM[i+1],work_TT[i+1]#亲代2

                    if np.random.rand()<self.p1:
                        W1,W2 = self.Crossover_Operation(W1,W2)
                        M1,M2 = self.Crossover_Machine(M1,M2)
                    if np.random.rand()<self.p2:
                        M1,T1=self.Mutation_Machine(W1,M1,T1)
                        M2,T2=self.Mutation_Machine(W2,M2,T2)
                        W1 = self.Mutation_Operation(W1,M1,T1)
                        W2 = self.Mutation_Operation(W2,M2,T2)
                    

                    C_finish,_,_=self.caculate1(W1,M1,T1,error_M_T)
                    work_job3[i]=W1  #更新工序编码
                    answer1.append(C_finish)
                    work_M3[i]=M1
                    work_T3[i]=T1

                    C_finish,_,_=self.caculate1(W2,M2,T2,error_M_T)
                    work_job3[i+1]=W2  #更新工序编码
                    answer1.append(C_finish)
                    work_M3[i+1]=M2
                    work_T3[i+1]=T2

                work_job2,work_M2,work_T2=np.vstack((work_job,work_job3)),np.vstack((work_M,work_M3)),np.vstack((work_T,work_T3))
                _, first_indices= np.unique(np.hstack((work_job2,work_M2)), axis = 0,return_index = True)
                work_job2, work_M2, work_T2 = work_job2[first_indices], work_M2[first_indices], work_T2[first_indices]

                answer2=answer+answer1#亲代子代结合比较
                answer2 = np.array(answer2)[first_indices].tolist()
                best_idx=np.array(answer2).argsort()[0:self.popsize] #前self.popsize个最好的个体，并将其作为下一次的亲代
                work_job,work_M,work_T=work_job2[best_idx],work_M2[best_idx],work_T2[best_idx]
            
                answer=np.array(answer2)[best_idx].tolist()
                best_index=answer2.index(min(answer2))             #找到最小完工时间的个体

                result.append([gen+1,min(answer2)])#记录每一次迭代的最优个体
            job, machine, machine_time= work_job2[best_index],work_M2[best_index],work_T2[best_index]
            make_span,list_S,list_W=self.caculate1(job,machine,machine_time,error_M_T)

        end_range = float(make_span + arrive_time[-1])
        data_list += [(machine[i],list_S[i]+arrive_time[-1],list_S[i]+list_W[i]+arrive_time[-1],job[i]) for i in range(len(job))]
        # print(data_list)
        self.draw_Gantt(data_list, arrive_time,initial_time,end_range)

        

   