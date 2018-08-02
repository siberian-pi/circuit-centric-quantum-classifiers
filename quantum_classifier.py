"""
根据circuit centric quantum classifier论文实现
的quantum_classifier

@jjq，2018.08.02

"""


import math
import numpy as np
import random

cos=math.cos
sin=math.sin
pi=math.pi
repeat_times=1000
learning_rate=0.01

def rand():#产生随机角度
    return random.random()*2*pi

def gcd(n,r):
    if(r==0):
        return n
    n=n-int(n/r)*r
    return gcd(r,n)
    


class Quantum_Classifier:
    #属性
    initial_state=[]#数组。初态，测量时会重用
    state=[]#数组。终态，存放振幅
    qubit_number=0#整数。qubit总数
    state_number=0#2**qubit_number
    circuit_size=0#整数。门的数目
    quantum_circuit_list=[]#数组，按顺序存放[门种类，[位置]，[参数]]
    bias=0

    #方法
    #定义电路,直接通过quantum_circuit_list定义；或者使用u,cp定义
    #python不支持函数重载...使用默认值
        

    def __init__(self,quantum_circuit_list_input=[],qnum=0,cnum=0):
        #检查输入是否合法
        #允许使用的门u(phi,beta,gamma),cp(phi)
        #print(quantum_circuit_list_input,qnum,cnum)
        if(not(quantum_circuit_list_input or qnum or cnum)):
            #print("error")
            return
        for gate_info in quantum_circuit_list_input:
            if(gate_info[0]=='u' and len(gate_info[1])==1 and len(gate_info[2])==4):
                pass
            else:
                if(gate_info[0]=='cp' and len(gate_info[1])==2 and len(gate_info[2])==1):
                    pass
                else:
                    print("电路输入不合法，只允许使用'u'或'cp")
                    return
        self.quantum_circuit_list=quantum_circuit_list_input
        self.qubit_number=qnum
        self.state_number=2**qnum
        self.circuit_size=cnum
        self.bias=rand()

    def u(self,phi,alpha,beta,gamma,position):
        self.quantum_circuit_list.append(['u',[position],[phi,alpha,beta,gamma]])
        self.circuit_size+=1
        return

    def cp(self,phi,position1,position2):
        self.quantum_circuit_list.append(['cp',[position1,position2],[phi]])
        self.circuit_size+=1
        return
    
    def init_state(self,some_state):
        some_state[0]*=(1+0j)#把类型变为复数
        self.initial_state=np.array(some_state)
        self.state=self.initial_state.copy()
        return
    
    def init_qnum(self,qnum):
        self.qubit_number=qnum
        self.state_number=2**qnum
        return
    
    def init_circuit(self,r=1):#一种标准的结构。包括参数初始化
        if(self.qubit_number==0):
            print("error.未设置qubit数目!")
            return
        if(self.quantum_circuit_list!=[]):
            print("error.circuit已存在!")
        n=self.qubit_number
        for i in range(n):
            self.quantum_circuit_list.append(['u',[i],[rand(),rand(),rand()]])

        self.quantum_circuit_list.append(['cp',[0,n-1],[rand()]])
        self.quantum_circuit_list.append(['u',[n-1],[rand(),rand(),rand()]])

        for i in range(self.qubit_number-1)[::-1]:
            self.quantum_circuit_list.append(['cp',[i,i-1],[rand()]])
            self.quantum_circuit_list.append(['u',[i],[rand(),rand(),rand()]])
            
        for i in range(self.qubit_number):
            self.quantum_circuit_list.append(['u',[i],[rand(),rand(),rand()]])

        m=gcd(n,r)
        for j in range(int(n/m)):
            self.quantum_circuit_list.append(['cp',[(j*r-r)%n,(j*r)%n],[rand()]])
            self.quantum_circuit_list.append(['u',[(j*r-r)%n],[rand(),rand(),rand()]])

        self.quantum_circuit_list.append(['u',[0],[rand(),rand(),rand()]])
        self.bias=rand()    
            
    def init_circuit_by_circuit(circuit):
        self.quantum_circuit_list=circuit

    #前向传播，计算quantum_circuit_list后的state
    def execute(self):
        if(not(self.check())):#检查初始化是否完成
           print("初始化未完成")
           return
        if(self.quantum_circuit_list==[]):
            print("电路为空")
            return
        for gate_info in self.quantum_circuit_list:
            self.compute(gate_info)
        return

    def check(self):
        if(self.initial_state==[] or self.state==[]):
            return 0
        if(not(self.qubit_number or self.circuit_size or self.state_number or self.quantum_circuit_list)):
           return 0 
        return 1        

    def measure_position(self,index):        
        self.swap(0,index)
        sum1=sum([ x.real**2+x.imag**2 for x in self.state[:int(self.state_number/2)]])
        sum2=sum([ x.real**2+x.imag**2 for x in self.state[int(self.state_number/2):]])
        self.swap(0,index)
        x=random.random()
        if(x>sum1/(sum1+sum2)):
            return '1'
        else:
            return '0'
    def measure(self,index,repeat_times):
        count=0
        for i in range(repeat_times):
            if(self.measure_position(index)=='0'):
                count+=1
        return {'0':count,'1':repeat_times-count}
            
    #辅助计算。改进，类似快速幂运算？
    def swap(self,position1,position2):#从0开始
        if(position1==position2):
            return
        n=self.qubit_number
        position1=n-position1-1#从0开始
        position2=n-position2-1
        if(position1<position2):
            temp=position1
            position1=position2
            position2=temp
        state_number=self.state_number
        for position in range(state_number):
            n1=2**position1
            temp_1=int(position/(n1))%2
            if(temp_1==1):#计算要交换的位置
                n2=2**position2
                temp_2=int(position/(n2))%2
                if(temp_2==0):
                    temp=self.state[position]
                    self.state[position]=self.state[position-n1+n2]
                    self.state[position-n1+n2]=temp
        return

    def compute_bit(self,position,index):
        index=self.qubit_number-index-1
        return int(position/(2**index))%2
    
    def compute_u(self,phi,alpha,beta,gamma):#在第0位使用u
        a=(cos(phi+beta)+sin(phi+beta)*1j)*cos(alpha)
        b=(cos(phi+gamma)+sin(phi+gamma)*1j)*sin(alpha)
        c=-(cos(phi-gamma)+sin(phi-gamma)*1j)*sin(alpha)
        d=(cos(phi-beta)+sin(phi-beta)*1j)*cos(alpha)
        temp=self.state.copy()
        m=int(self.state_number/2)
        self.state[:m]=a*temp[:m]+b*temp[m:]
        self.state[m:]=c*temp[:m]+d*temp[m:]
        return

    #测试用
    def get_amplitude(self,basis):
        return (self.state[basis])
        
    def get_state(self):
        return self.state
        
    def compute(self,gate_info):
        if(gate_info[0]=='cp'):
            control,target=gate_info[1]
            for position in range(self.state_number):
                temp=self.compute_bit(position,control)
                if(temp==1):
                    [phi]=gate_info[2]
                    temp=self.compute_bit(position,target)
                    if(temp==1):
                        self.state[position]*=(cos(phi)+sin(phi)*1j)        
        if(gate_info[0]=='u'):
            [position]=gate_info[1]
            phi,alpha,beta,gamma=gate_info[2]
            self.swap(0,position)
            self.compute_u(phi,alpha,beta,gamma)
            self.swap(0,position)
        return
            
    #反向传播
    def train_mnist(self,train_file,label_file,batch_size=1):

        binfile_train=open(train_file,'rb')
        buf_train=binfile_train.read()

        binfile_label=open(label_file,'rb')
        buf_label=binfile_label.read()

        index_train=0
        magic,img_num,numRows,numColums=struct.unpack_from( '>IIII', buf_train, index_train )
        index_train += struct.calcsize('>IIII')

        index_label=0
        struct.unpack_from('>II',buf_label,index_label) #按照MNIST数据集的说明，读取两个unsigned int32
        index_label+=struct.calcsize('>II')

        count=0
        train_data=[]
        
        #初始化电路，qubit_number
        if(self.quantum_circuit_list==[]):
            print("根据标准方式初始化circuit")
            self.init_circuit()
        self.init_qnum(10)    

        for i in range(img_num):
            
            label=struct.unpack_from('>1B',buf_label,index_label)[0]
            index_label+=struct.calcsize('>1B')        
            im=struct.unpack_from( '>784B', buf_train, index_train )
            index_train += struct.calcsize('>784B')

            ##################################################################
            #如果希望计算简单，这里可以先将im变为0,1
            ##################################################################
            train_data.append([im,label])
            count+=1
            if(count==batch_size):
                self.update(train_data,batch_size)
                count=0
                train_data=[]
        return

    def update(self,train_data):#采用10个qubit,im编码补全
        pass

        
        
    
        
    
        

    
"""
测试
swap
compute H,cp,组合
"""
