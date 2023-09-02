import numpy as np
from data_solve import data_deal
import random 
import matplotlib.pyplot as plt 
#plt.rcParams['font.sans-serif'] = ['STSong'] 
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


class FJSP(): #! initialization 并未用到全局选择和局部选择 反而 全是随机选择
	def __init__(self,job_num,machine_num,parm_data):
		self.job_num=job_num     			#工件数
		self.machine_num=machine_num		#机器数
		self.Tmachine,self.Tmachinetime,self.tdx,self.work,self.tom,self.machines=parm_data[0],parm_data[1],parm_data[2],parm_data[3],parm_data[4],parm_data[5]
	def axis(self):
		index=['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12',
		'M13','M14','M15','M16','M17','M18','M19','M20']
		scale_ls,index_ls=[],[]   
		for i in range(self.machine_num):
			scale_ls.append(i+1)
			index_ls.append(index[i])
		return index_ls,scale_ls  #返回坐标轴信息，按照工件数返回，最多画20个机器，需要在后面添加
	
	# def creat_job(self,job,machine,machine_time,index): #初始化 重调度 工序随机选择机器
	# 	count=np.zeros((self.job_num,1),dtype=np.int)-1
	# 	job1,job2=job[0,:index+1].tolist(),job[0,index+1:] #按第一行划分那些固定那些要改动
	# 	np.random.shuffle(job2) #随机安排operation
	# 	work=job1+job2.tolist()
	# 	machine1,machine_time1=machine.copy(),machine_time.copy()
		
	# 	for i in range(job.shape[1]):#列向量即工序的长度（所有工序的个数）
	# 		signal=int(job[0,i])#扫到的工件
	# 		count[signal,0]+=1 #扫过的工件次数-1
	# 		if(i>index):
	# 			highs=self.tom[signal][count[signal,0]]
	# 			lows=self.tom[signal][count[signal,0]]-self.tdx[signal][count[signal,0]]
	# 			n_machine=self.Tmachine[signal,lows:highs].tolist() #可选机器的范围
	# 			n_time=self.Tmachinetime[signal,lows:highs].tolist()
													 
	# 			index1=np.random.randint(0,len(n_time),1)#随机选取机器
	# 			machine1[0,i]=n_machine[index1[0]]
	# 			machine_time1[0,i]=n_time[index1[0]]
	# 	return np.array([work]),machine1,machine_time1
	
	def caculate(self,job,machine,machine_time):
		jobtime=np.zeros((1,self.job_num))        
		tmm=np.zeros((1,self.machine_num))   			
		tmmw=np.zeros((1,self.machine_num))			
		startime=0
		list_M,list_S,list_W=[],[],[]
		count=np.zeros((1,self.job_num),dtype=np.int)
		for i in range(job.shape[1]):
			svg=int(job[0,i])
			sig=int(machine[0,i])-1

			startime=max(jobtime[0,svg],tmm[0,sig])   	
			tmm[0,sig]=startime+machine_time[0,i]
			jobtime[0,svg]=startime+machine_time[0,i]
	
			# list_M.append(sig+1)
			list_S.append(startime)
			list_W.append(machine_time[0,i])
			count[0,svg]+=1
				       
		tmax=np.argmax(tmm[0])+1		#结束最晚的机器
		C_finish=max(tmm[0])			#最晚完工时间
		return C_finish,list_S,list_W,tmax
	
	def draw(self,job,C_finish,list_M,list_S,list_W,tmax,signal,error_M,error_S,error_T):#画图   
		figure,ax=plt.subplots()
		count=np.zeros((1,self.job_num))
		for i in range(job.shape[1]):  #每一道工序画一个小框
			count[0,int(job[0,i])-1]+=1
			plt.bar(x=list_S[i], bottom=list_M[i], height=0.5, width=list_W[i], orientation="horizontal",color='white',edgecolor='black')
			plt.text(list_S[i]+list_W[i]/32,list_M[i], '%.0f' % (job[0,i]+1),color='black',fontsize=10,weight='bold')#12是矩形框里字体的大小，可修改
		plt.plot([C_finish,C_finish],[0,tmax],c='black',linestyle='-.',label='完工时间=%.1f'% (C_finish))#用虚线画出最晚完工时间
		
		font1={'weight':'bold','size':22}#汉字字体大小，可以修改
		plt.xlabel("加工时间",font1)
		plt.title("甘特图",font1)
		plt.ylabel("机器",font1)
		if(signal>0):
			plt.plot([error_S,error_S],[0,self.machine_num+1],c='black',linestyle='--',label='故障开始时间=%.1f'% (error_S))
			plt.plot([error_S+error_T,error_S+error_T],[0,error_M],c='black',linestyle=':',label='故障结束时间=%.1f'% (error_S+error_T))

	# 	scale_ls,index_ls=self.axis()
	# 	plt.yticks(index_ls,scale_ls)
	# 	plt.axis([0,C_finish*1.1,0,self.machine_num+1])
	# 	plt.tick_params(labelsize = 22)#坐标轴刻度字体大小，可以修改
	# 	labels=ax.get_xticklabels()
	# 	[label.set_fontname('SimHei')for label in labels]
	# 	plt.legend(prop={'family' : ['SimHei'], 'size'   : 16})#标签字体大小，可以修改
	# 	plt.xlabel("加工时间",font1)






	def caculate1(self,job,machine,machine_time,error_M,error_S,error_T):
		jobtime=np.zeros((1,self.job_num))      #上一个工序的结束时间
		tmm=np.zeros((1,self.machine_num))   	#tmm机器可使用的最早时间		
		# tmmw=np.zeros((1,self.machine_num))			
		startime=0
		list_M,list_S,list_W=[],[],[]
		count=np.zeros((1,self.job_num),dtype=np.int)
		for i in range(job.shape[1]):#遍历工序单（基因）
			svg=int(job[0,i])
			sig=int(machine[0,i])-1

			startime=max(jobtime[0,svg],tmm[0,sig])
			if(startime<error_S)and(startime+machine_time[0,i]>error_S)and(error_M==sig+1):#出故障的机器、和对应正在进行的工序的位置
				startime=error_S+error_T   #处理时间变长
			tmm[0,sig]=startime+machine_time[0,i]
			jobtime[0,svg]=startime+machine_time[0,i]
			
			list_M.append(sig+1)
			list_S.append(startime)
			list_W.append(machine_time[0,i])
			count[0,svg]+=1
				       
		tmax=np.argmax(tmm[0])+1		#结束最晚的机器
		C_finish=max(tmm[0])			#最晚完工时间
		return C_finish,list_M,list_S,list_W,tmax