import tensorflow as tf
from tensorflow.contrib import slim
import matplotlib.pyplot as plt

'''
在slim中模型变量在训练过程中不断被训练和调参，非模型变量（常规变量）
是指那些在训练和评估中用到的但是在预测阶段没有用到的变量
'''
weights = slim.model_variable('weights', shape=[10, 10, 3, 3],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05), device='/CPU:0')
model_variables = slim.get_model_variables()
# 常规变量
my_var = slim.variable("my_var", shape=[2, 3], initializer=tf.zeros_initializer())
all_variables = slim.get_variables()  # 获取模型变量+常规变量
'''
slim常用的函数：
slim.bias_add() 加上偏置
slim.batch_norm() 归一化
slim.conv2d() 二维卷积
slim.conv2d_in_plane() 
slim.conv2d_transpose() 反卷积
slim.fully_connected() 全连接
slim.avg_pool2d() 二维平均池化
slim.dropout() 
slim.flatten() 展为一维
slim.max_pool2d() 二维最大池化
slim.one_hot_encoding() onehot编码
slim.separable_conv2d() 可分离卷积
slim.unit_norm() 单位归一化
slim.losses.softmax_cross_entropy() 交叉熵损失 
slim.losses.sum_of_squares() 均方误差
'''
input = 66
net = slim.repeat(input, 3, slim.conv2d, 256, [3, 3], scope="conv3")
net = slim.max_pool2d(net, [2, 2], scope="pool2")
# 3次调用slim.fully_connected()，每次函数的输出传递给下一次的输入，
# 隐藏层的单元数分别为32、64和128
slim.stack(input, slim.fully_connected, [32, 64, 128], scope="fc")

#通过slim.arg_scope()函数使2个卷积层和1个全连接层共享相同的参数，可嵌套使用
with slim.arg_scope([slim.conv2d,slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    with slim.arg_scope([slim.conv2d],stride=1,padding="SAME"): #只设置slim.conv2d的默认参数
        net=slim.conv2d(input,64,[11,11],scope="conv1")
        net=slim.conv2d(net,128,[11,11],padding="VALID",scope="conv2")
        net=slim.fully_connected(net,1000,activation_fn=None,scope="fc")

total_loss=slim.losses.get_total_loss() #获取总损失
learning_rate=0.1
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train_op=slim.learning.create_train_op(total_loss,optimizer)
logdir=r"./checkpoint/" #checkpoint和event文件的存放目录
slim.learning.train(train_op,logdir,number_of_steps=1000, #更新次数
                    save_summaries_secs=300, #每5分钟计算一次summaries
                    save_interval_secs=600) #每10分钟保存一次checkpoint文件