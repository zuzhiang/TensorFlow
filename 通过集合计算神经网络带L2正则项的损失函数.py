'''
以下代码表示的是一个5层(包括输入层)的全连接网络。
'''
import tensorflow as tf

#获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为losses的集合中
def get_weight(shape,lambda_):
    weight=tf.Variable(tf.random.normal(shape),dtype=tf.float32)
    # tf.add_to_collection()的两个参数分别为集合名以及加入到这个集合的内容
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lambda_)(weight))
    #返回的weight是未L2正则化后的
    return weight

if __name__=="__main__":
    x=tf.placeholder(tf.float32,shape=(None,2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1)) #真实值
    batch_size=8
    #每一层网络中节点的个数
    layer_dimension=[2,10,10,10,1]
    #神经网络的层数
    n_layers=len(layer_dimension)

    #该变量维护前向传播时最深层的节点，开始的时候就是输入层
    cur_layer=x
    #当前层的节点个数
    in_dimension=layer_dimension[0]

    #通过循环来生成5层的全连接神经网络
    for i in range(1,n_layers):
        #下一层的节点个数
        out_dimension=layer_dimension[i]
        #生成当前层权重的变量，并将其L2正则化损失加入到计算图的集合中
        weight=get_weight([in_dimension,out_dimension],0.001)
        bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))
        #使用relu激活函数
        cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
        #进行下一层之前将下一层的节点个数更新为当前层节点个数
        in_dimension=layer_dimension[i]

    #经过循环后cur_layer即输出层，也就是预测值
    mse_loss=tf.reduce_mean(tf.square(y_-cur_layer))

    #将均方误差损失函数加入损失集合
    tf.add_to_collection("losses",mse_loss)
    #tf.get_collection()返回一个集合中所有元素的列表
    #tf.add_n()可实现列表中所有元素相加，元素可能是张量， 返回值和输入值shape相同
    loss=tf.add_n(tf.get_collection("losses"))
    print("end")