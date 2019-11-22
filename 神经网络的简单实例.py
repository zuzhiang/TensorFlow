'''
以下代码实现了一个神经网络的整个训练过程，该神经网络中输入层
为实体的两个特征，记为x1，x2；有一个隐藏层（神经网络的第一层），
该层有3个神经元（节点），记为a1，a2，a3；输入层有1个神经元，记
为y。输入层和隐藏层以及隐藏层和输出层直接都是采用的全连接，分别
有2*3=6个参数和3*1=3个参数，记为Wij。
'''
'''
神经网络的训练可以分为三个步骤：
1.定义神经网络的结构和前向传播的输出结果
2.定义损失函数以及选择反向传播优化的方法
3.生成会话（tf.Session）并且在训练集上反复进行反向传播优化算法
'''
import tensorflow as tf
from numpy.random import RandomState

#定义训练数据batch的大小
batch_size=8

w1=tf.Variable(tf.random.normal((2,3),stddev=1,seed=1))
w2=tf.Variable(tf.random.normal((3,1),stddev=1,seed=1))


#在shape的一个维度上使用None可以方便使用不同的batch大小，
x=tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_=tf.placeholder(tf.float32,shape=(None,1),name="y-input")

#前向传播
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#定义损失函数和反向传播过程
'''
 使用sigmoid函数将y转化为0~1之间的值，转换后y表示预测是
 正样本的概率，1-y表示预测是负样本的概率。
'''
y=tf.sigmoid(y)
'''
定义预测值和真实值之间的交叉熵损失
交叉熵损失函数为：loss=-[y*log(y^)+(1-y)*log(1-y^)]
其中y是样本的真实值，y^是样本的预测值
'''
cross_entropy=-tf.reduce_mean(
    y_*tf.log(tf.clip_by_value(y,1e-10,1.0))
    +(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
'''
tf.clip_by_value的作用是对tesor数据进行截断，使其在一定范围内。
其参数有：
t：张量名
clip_value_min：最小值 
clip_value_max：最大值
name=None：名称，默认无
'''
learning_rate=0.001 #学习率
#定义反向传播的优化方法
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
'''
常用的优化方法有三种：
1  tf.train.GradientDescentOptimizer 梯度下降
2  tf.train.AdamOptimizer 
3  tf.train.MomentumOptimizer 动量优化
'''

#随机生成一个模拟数据集
rdm=RandomState(1)
datase_size=128
X=rdm.rand(datase_size,2)
print("X:\n",X)
'''
所有满足 x1+x2<1的样本都被认为是正样本，用1表示，
反之为负样本，用0表示。
'''
Y=[[int(x1+x2<1)] for (x1,x2)in X]

with tf.Session() as sess:
    #以下两行代码的作用同tf.global_variables_initializer().run()
    init_op=tf.global_variables_initializer()
    sess.run(init_op) #初始化变量
    print("训练前的参数：")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))

    steps=5000 #训练轮数
    for i in range(steps):
        start=(i*batch_size)%batch_size #每次选取batch_size个样本进行训练
        end=min(start+batch_size,datase_size)

        #通过选取的样本训练神经网络并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            #每个一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))
    print("训练后的参数：")
    print("w1:\n",sess.run(w1))
    print("w2:\n", sess.run(w2))
