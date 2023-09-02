import numpy as np
from fjsp import FJSP
from data_solve import data_deal
import matplotlib.pyplot as plt 
from ga import GA
#plt.rcParams['font.sans-serif'] = ['STSong'] 
from matplotlib.pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

job_num = 10
machine_num  =6
###############处理表格信息###############
oj=data_deal(job_num,machine_num)               #工件数，机器数
Tmachine,Tmachinetime,tdx,work,tom,machines=oj.cacu()#Tmachine 把M01中的机器提取出来[[]],Tmachinetime[[]]对应处理时间
# #tdx[[]]每个工件的工序的可用机器数，work 按tdx生成job编号[0,0,0,0,0,0,1,1,...]  machines[6, 5, 5, 5, 6, 6, 5, 5, 6, 6]每个工件包含的工序数 tom累加 

parm_data=[Tmachine,Tmachinetime,tdx,work,tom,machines]
# to=FJSP(job_num,machine_num,parm_data)#仿真环境
##############GA生成结果##################
# init=GA(20,10,to,0.8,0.2,parm_data,machine_num) 
# job,machine,machine_time = init.ga_total(...)
#############插入新工件（生成新的模块信息）###########
#6 2 3 4 6 2 1 1 2 3 3 4 2 6 6 6 1 2 6 3 6 5 2 6 1 1 2 1 3 4 2
#new_operation = {arrive_time:[num_工序，[Tmachine],[Tmachinetime],tdx,tom]}
new_operation = {7:[6,[3,6,1,3,2,6,2,6,2,1,1,4],[4,2,2,4,6,6,6,5,6,1,3,2],[2,1,3,1,3,2],[2,3,6,7,10,12]],20:[6,[3,6,1,3,2,6,2,6,2,1,1,4],[4,2,2,4,6,6,6,5,6,1,3,2],[2,1,3,1,3,2],[2,3,6,7,10,12]]}
#TODO when time = 20 
update = GA(20,20,0.8,0.2,parm_data,machine_num)
update.ga_total(new_operation)


