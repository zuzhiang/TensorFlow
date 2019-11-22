#以下代码定义了前向传播过程及神经网络中的参数
import tensorflow as tf

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
#以下为LeNet独有的常量
IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10
CONV1_DEEP=32 #第一层卷积层的深度
CONV1_SIZE=5 #第一层卷积层的尺寸
CONV2_DEEP=64
CONV2_SIZE=5
FC_SIZE=512 #全连接层的节点个数


#以下为全连接神经网络中前向传播的实现代码
#获取神经网络中的变量
def get_weight_variable(shape,regularizer):
    weights=tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    #当给出了正则化生成函数时，将当前变量的正则化损失加入到自定义集合losses中
    if regularizer!=None:
        tf.add_to_collection("losses",regularizer(weights))
    return weights #返回的weight是未正则化后的

#定义神经网络的前向传播过程
def inference(input_tensor,regularizer):
    with tf.variable_scope("layer1"):
        weights=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases=tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases) #用relu做激活函数

    with tf.variable_scope("layer2"):
        weights=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer1,weights)+biases
    return layer2

'''
def inference(input_tensor,train,regularizer):
    with tf.variable_scope("layer1-conv1"):
        conv1_weights=tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

        with tf.name_scope("layer2-pool1"):
            pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        with tf.variable_scope("layer3-conv2"):
            conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

            conv2 = tf.nn.conv2d(input_tensor, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        with tf.name_scope("layer2-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        pool_shape=pool2.get_shape().as_list()
        nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]

        reshaped=tf.reshape(pool2,[pool_shape[0],nodes])

        with tf.variable_scope("layer5-fc1"):
            fc1_weights=tf.get_variable("weight",[nodes,FC_SIZE],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer!=None:
                tf.add_to_collection("losses",regularizer(fc1_weights))
            fc1_biases=tf.get_variable("bias",[FC_SIZE],initializer=tf.constant(0.1))
            fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
            if train:
                fc1=tf.nn.dropout(fc1,0.5)

        with tf.variable_scope("layer6-fc1"):
            fc2_weights=tf.get_variable("weight",[FC_SIZE,NUM_LABELS],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer!=None:
                tf.add_to_collection("losses",regularizer(fc2_weights))
            fc2_biases=tf.get_variable("bias",[NUM_LABELS],initializer=tf.constant(0.1))
            logit=tf.matmul(fc1,fc2_weights)+fc2_biases

        return logit
'''