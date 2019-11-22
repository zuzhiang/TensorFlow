#测试模型的好坏
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

#每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
        # 采用LeNet时的输入变量格式，为一个四维矩阵
        '''
        x = tf.placeholder(tf.float32, [mnist_train.BATCH_SIZE,  # 一个batch中样例的个数
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.NUM_CHANNELS],  # 图片的深度
                           name="x-input")
        '''
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")
        #LeNet中验证集上的输入数据
        '''
        validate_x = np.reshape(mnist.validation.images, (mnist_train.BATCH_SIZE,  # 一个batch中样例的个数
                                                  mnist_inference.IMAGE_SIZE,
                                                  mnist_inference.IMAGE_SIZE,
                                                  mnist_inference.NUM_CHANNELS))  # 图片的深度
        '''
        # 验证集
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        #前向传播过程
        y=mnist_inference.inference(x,None)
        #计算在验证集上的正确率
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #通过变量重命名的方式来加载模型，这样在前向传播过程中
        #就不必调用求滑动平均的函数来获取平均值了
        variable_averages=tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)

        sess=tf.InteractiveSession()
        #tf.train.get_checkpoint_state函数会通过checkpoint
        # 文件自动找到目录中最新模型的文件名
        ckpt=tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            #加载模型
            saver.restore(sess,ckpt.model_checkpoint_path)
            #通过文件名得到模型保存时迭代的次数
            global_step=ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
            accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
            print("After %s traning steps, validation accuracy=%g"%(global_step,accuracy_score))
        else:
            print("No checkpoint file found.")
            return

def main(argv=None):
    mnist=input_data.read_data_sets(r"C:\Users\zuzhiang\PycharmProjects\untitled2\TensorFlow\MNIST", one_hot=True)
    evaluate(mnist)

if __name__=="__main__":
    tf.app.run()
    print("end")
