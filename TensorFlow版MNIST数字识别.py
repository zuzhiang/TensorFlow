'''
以下代码在MNIST手写数字识别数据集的基础上，构建了一个
只有一个隐藏层的神经网络，对其进行分类。

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#输入层节点个数，即图片的像素数（28*28）
INPUT_NODE = 784
#输出层节点个数，也就是类别个数
OUTPUT_NODE = 10
#隐藏层节点个数，该网络只有一个隐藏层
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8 #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率
REGULARIZATION_RATE = 0.0001 #正则项在损失函数中的系数
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率


#以下函数实现了前向传播过程，第二个参数表示是否是滑动平均类
def inference(input_tensor, avg_class,reuse=False):
    #根据传进来的reuse值来判断是创建新变量还是使用已经创建好的
    with tf.variable_scope("layer",reuse=reuse):
        weights1=tf.get_variable("weights1",[INPUT_NODE,LAYER1_NODE],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases1=tf.get_variable("biases1",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        weights2 = tf.get_variable("weights2", [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases2 = tf.get_variable("biases2", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))

    #没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        '''
        因预测时使用的是不同类别对应节点的输出值的相对大小，有没有
        softmax层对最后分类的结果没有影响，故可以不加softmax层
        '''
        return tf.matmul(layer1, weights2) + biases2,weights1,weights2

    else:
        #使用avg_class.average()函数来计算滑动平均之后变量weight1/2的值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))
                            + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) \
               + avg_class.average(biases2)


def train(mnist):
    #[None,INPUT_NODE]可以理解为有任意组为输入向量大小的训练数据
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    '''
    计算在当前参数下神经网络前向传播的结果。因用于计算滑动平均的类
    为None，所以函数不会使用参数的滑动平均值
    '''
    y,weights1,weights2= inference(x, None)
    #训练轮数，一般设为不可训练的参数
    global_step = tf.Variable(0, trainable=False)
    #初始化滑动平均类
    '''
    tf.train.ExponentialMovingAverage()实现滑动平均模型，其参数有：
    decay:衰减率可以控制模型更新的速度，其值越大模型越稳定，
    num_updates:使用的动态的衰减率(decay)，其计算公式为：
        decay=min{decay,(1+num_updates)/(10+num_updates)}
    
    该模型会对每个变量维护一个影子变量，影子变量的初始值为原变量的值，
    并通过以下公式进行更新：
        shadow_variable = decay * shadow_variable + (1 - decay) * variable
    
    滑动平均模型可以让神经网络的参数平均滑动（参数变化比较平滑），
    其返回值是一个滑动平均的类。相当于用影子变量去拟合原变量，而
    影子变量的值就是滑动平均值，它可以看作是 1/(1 - decay) 个样本
    数据的平均值，而如果decay的值是固定的话，在一开始时，因为数据
    量达不到decay的大小，所以会出现较大的偏差，所以可以用 
    num_updates 来动态的设置衰减率。
    '''
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)
    '''
    在所有神经网络的参数上使用滑动平均，tf.trainable_variables()返回所有可训练变量
    
    定义一个更新变量滑动平均的操作，这里需要给定一个列表（所有可训练变量），每次
    执行该操作时，这个列表中的变量都会被更新
    '''
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #计算使用了滑动平均后的前向传播结果
    average_y = inference(x, variable_averages,True) #因是第二次调用，故参数已经创建过了
    '''
    使用交叉熵来作为损失函数
    
    使用tf.nn.sparse_softmax_cross_entropy_with_logits()函数
    可将输出层的输出先经过softmax函数，再计算其交叉熵，当分类
    问题只有一个答案时，该函数可加快计算。需要先经过softmax的
    原因是神经网络的原始输出值可能不是一个概率分布，无法直接
    计算交叉熵。其参数为：
    _sentinel=None:
    labels:样本的真实标签
    logits:神经网络输出层的输出
    name:名称
    '''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                   labels=tf.argmax(y_, 1))
    #计算当前batch中所有样例交叉熵的均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #tf.contrib.layers.l1_regularizer()返回的是一个可以计算参数L1正则项的值的函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失=交叉熵损失 + 正则化损失
    loss = cross_entropy_mean + regularization
    '''
    设置指数衰减的学习率
    
    tf.train.exponential_decay()其功能为采用指数衰减的学习率，学习率的计算公式如下：
    decayed_learning_rate=learning_rate * decay_rate ** (global_step / decay_steps) 

    其参数如下：
    learning_rate：初始学习率
    global_step：用于衰减计算的全局步骤，喂一次batch_size的数据计算一次global_step
    decay_steps：衰减速率，非负数，每隔decay_steps次更新一次learning_rate值
    decay_rate：衰减系数，
    staircase： 为True时，学习率呈阶梯函数状是离散的，反之是连续的
    name：名称
    '''
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE,LEARNING_RATE_DECAY)
    #用梯度下降算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                                           global_step=global_step)
    '''
    在训练神经网络时，每过一遍数据就要通过反向传播来更新网络的参数，
    并且更新每一个参数的滑动平均值，以下两行代码的作用等同于：
    train_op=tf.group(train_step,variable_vaerages_op)
    
    control_dependencies(control_inputs)返回一个上下文管理器，它指定
    了控制依赖项，其参数为一个列表。以下代码的含义是只有在train_step
    和variable_averages_op运行之后train_op才会运行
    
    tf.no_op()表示执行完 train_step, variable_averages_op 操作之后什么都不做
    '''
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")
    '''
    下一行代码可判断预测的是否正确
    
    tf.argmax(input,axis)可根据axis取值的不同返回每行或者每列最大值的索引
    
    tf.argmax(average_y,1)可以计算每个样例的预测答案，average_y是
    batch_size * 10的二维数组，第二个参数表示在每一行中取最大值，
    该函数输出为 batch_size * 1的数组
    
    tf.equal(t1,t2)是判断两个张量的每一维是否相等
    '''
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    #一组数据上的正确率
    # tf.cast(x, dtype, name=None)可将x转化为指定的数据类型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #初始会话，并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #准备验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        #准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        #迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                #以下代码的含义是将validate_feed喂给神经网络，并获取accuracy（准确率）的值
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average "
                      "model is %g, test accuracy using average model is %g" % (i, validate_acc,test_acc))
            # 每轮从训练集中选取batch_size个训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            #在训练集上进行训练
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        #训练结束后在测试集上检测模型的正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After $d training step(s), test accuracy using average model "
              "is %g" % (TRAINING_STEPS, test_acc))


def main(argv=None):
    # 通过input_data.read_data_sets()载入MNIST数据集，并将其划分为train、validation和test三部分
    mnist = input_data.read_data_sets(r"C:\Users\zuzhiang\PycharmProjects\untitled2\TensorFlow\MNIST", one_hot=True)

    print("Training data size: ", mnist.train.num_examples)
    print("Validating data size: ", mnist.validation.num_examples)
    print("Testing data size: ", mnist.test.num_examples)
    print("Example training data: \n", mnist.train.images[0])
    print("Example training data label: ", mnist.train.labels[0])

    train(mnist)


if __name__ == "__main__":
    #以下函数会自动调用上面定义的main函数
    tf.app.run()
