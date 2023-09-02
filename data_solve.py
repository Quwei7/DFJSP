import numpy as np 

class data_deal:
	def __init__(self,job_num,machine_num):
		self.job_num=job_num
		self.machine_num=machine_num
	def read(self):#List[List[int]]
		f=open('./MK01.txt')
		f1=f.readlines()
		c,count=[],0
		for line in f1:
			t1=line.strip('\n')
			if(count>0):
				sig=0
				cc=[]
				for j in range(len(t1)):
					if(t1[j]==' '):
						cc.append(int(t1[j-sig:j]))
						sig=0
					if(t1[j]!=' '):
						sig+=1
					if(j==len(t1)-1):
						cc.append(int(t1[j]))
						sig=0
				c.append(cc)
			count+=1
		return c 
	def translate(self,tr1):
		sigdex,mac,mact,sdx=[],[],[],[]
		sigal=tr1[0]
		tr1=tr1[1:len(tr1)+1]
		index=0
		for j in range(sigal):
			sig=tr1[index]
			sdx.append(sig)
			sigdex.append(index)
			index=index+1+2*sig
		for ij in range(sigal):
			del tr1[sigdex[ij]-ij]
		for ii in range(0,len(tr1)-1,2):
			mac.append(tr1[ii])
			mact.append(tr1[ii+1])
		return mac,mact,sdx
	
	def widthxx(self,strt):
		widthx=[]
		for i in range(self.job_num):
			mac,mact,sdx=self.translate(strt[i])
			siga=len(mac)
			widthx.append(siga)
		width=max(widthx)
		return width
	
	def tcaculate(self,strt):
		width=self.widthxx(strt)
		Tmachine,Tmachinetime=np.zeros((self.job_num,width)),np.zeros((self.job_num,width))
		tdx=[]
		for i in range(self.job_num):
			mac,mact,sdx=self.translate(strt[i])
			tdx.append(sdx)
			siga=len(mac)
			Tmachine[i,0:siga]=mac
			Tmachinetime[i,0:siga]=mact
		return Tmachine.astype(int).tolist(),Tmachinetime.tolist(),tdx
	
	def cacu(self):
		strt=self.read()
		Tmachine,Tmachinetime,tdx=self.tcaculate(strt)
		tom,work,machines=[],[],[]
		# to,tom,work,machines=0,[],[],[]
		for i in range(self.job_num):
			# to+=len(tdx[i])
			tim=[]
			for j in range(1,len(tdx[i])+1):
				tim.append(sum(tdx[i][0:j]))
				work.append(i)
			machines.append(len(tdx[i]))
			tom.append(tim)
			
		return Tmachine,Tmachinetime,tdx,work,tom,machines


# to=data_deal(10,6)

# c=to.read()
# Tmachine,Tmachinetime,tdx,work,tom,machines=to.cacu()
# print(machines)



