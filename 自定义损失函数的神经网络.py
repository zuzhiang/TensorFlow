'''
以下代码所表示的神经网络有两个输入节点和一个输出节点，没有隐藏层。
'''
import tensorflow as tf
from numpy.random import RandomState

batch_size=8

#tf.placeholder()用于定义过程，等到运行的时候再赋值，只有以下三个参数
x=tf.placeholder(tf.float32,shape=(None,2),name="x-input")
#回归问题一般只有一个输出节点
y_=tf.placeholder(tf.float32,shape=(None,1),name="y-input") #真实值

#前向传播过程
w1=tf.Variable(tf.random.normal((2,1),stddev=1,seed=1))
y=tf.matmul(x,w1) #预测值

#预测少了的惩罚度更大
loss_less=10 #预测少了的损失
loss_more=1 #预测多了的损失
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),
                            (y-y_)*loss_more,(y_-y)*loss_less))
'''
tf.reduce_mean()、tf.reduce_sum()和tf.reduce_max()有相似的用法
它们的作用分别是求张量在某个维度的平均值、总和以及最大值。
参数1：input_tensor表示输入张量
参数2：reduction_indices:在哪一维上求解，0为列，1为行，默认为None，此时表示对所有元素进行操作。
'''
learning_rate=0.001
#采用Adam优化方法对损失函数loss进行最小优化
train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)

rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2) #随机生成一个128*2的矩阵
print("X:\n",X)

Y=[[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

sess=tf.InteractiveSession() #创建并设置默认会话
#对所有变量进行初始化
init_op=tf.global_variables_initializer()
sess.run(init_op)
steps=5000
for i in range(steps):
    if i%100==0:
        print("step: ",i)
    start=(i*batch_size)%dataset_size
    end=min(start+batch_size,dataset_size)
    #x从X[start:end]中取值，y_从Y[start:end]中取值
    sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
    '''
    run(fetches,feed_dict=None,options=None,run_metadata=None)
    '''
print("最终学到的参数w1:\n",sess.run(w1))