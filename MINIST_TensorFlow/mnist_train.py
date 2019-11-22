#在训练集上对神经网络进行训练
import  os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH=r"C:\Users\zuzhiang\PycharmProjects\untitled2\TensorFlow\model"
MODEL_NAME="mnist.ckpt"

#训练神经网络
def train(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
    #采用LeNet时的输入变量格式，为一个四维矩阵
    '''
    x=tf.placeholder(tf.float32,[BATCH_SIZE, #一个batch中样例的个数
                                 mnist_inference.IMAGE_SIZE,
                                 mnist_inference.IMAGE_SIZE,
                                 mnist_inference.NUM_CHANNELS], #图片的深度
                     name="x-input")
    '''
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #前向传播
    y=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    #滑动平均模型
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    #softmax + 交叉熵损失
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection("losses"))
    #指数衰减的学习率
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    #梯度下降优化方法
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name="train")

    #初始化TensorFlow持久化类
    saver=tf.train.Saver()
    #定义默认会话，并对所有变量进行初始化
    sess=tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #在训练过程中不再测试模型在验证集上的表现
    for i in range(TRAINING_STEPS):
        #每次取batch_size个样本进行训练
        xs,ys=mnist.train.next_batch(BATCH_SIZE)
        # LeNet中需要将输入数据调整为四维矩阵才能传入sess.run()
        '''
        xs=np.reshape(xs,(BATCH_SIZE, #一个batch中样例的个数
                                 mnist_inference.IMAGE_SIZE,
                                 mnist_inference.IMAGE_SIZE,
                                 mnist_inference.NUM_CHANNELS)) #图片的深度
        '''
        train_op_,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
        #每1000轮保存一次
        if i%1000==0:
            print("After %d training steps, loss on training batch is %g."% (step,loss_value))
            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist=input_data.read_data_sets(r"C:\Users\zuzhiang\PycharmProjects\untitled2\TensorFlow\MNIST", one_hot=True)
    train(mnist)

if __name__=="__main__":
    tf.app.run()
    print("end")
