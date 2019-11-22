import time
import threading
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
Tensorflow 程序一般分为两个阶段，第一个阶段需要定义计算图中所有
的计算；第二个阶段为执行计算（通过会话实现）
'''
'''
np.random.rand()：
    以相等的概率返回一个区间[0,1)内的数，参数为矩阵的大小
np.random.randn():
    以正态分布的概率返回一个数，参数为矩阵的大小
'''
# 张量中没有保存计算的结果，而是保存了计算的过程，故在输出时只会输出对结果的一个引用
g = tf.Graph()  # 生成新的计算图，不同计算图上的张量和运算不会共享
g.as_default()  # 设置默认计算图
tf.get_default_graph()  # 获取当前默认图
# g.device("/gpu:0") #设置运行在第一个cpu上

'''
在计算图中可以通过集合来管理不同类别的资源。用 tf.add_to_collection 可以往一个或多个集合
中添加资源。用 tf.get_collection 可以获取一个集合中的所有资源。TensorFlow自带的常用集合
有：tf.GraphKeys.VARIABLES 是所有变量的集合；tf.GraphKeys.TRAINABLE_VARIABLES 是所有变量
的集合。
'''
# 在TensorFLow中张量中并不保存数字，而是保存数字的计算过程，并且保存张量的三大
# 属性：名字、维度和类型

'''
以下为TensorFlow中的常数生成函数
 
a = tf.zeros([2, 3], float)  # 产生大小为2*3的全0矩阵，指定大小时用(2,3)也可
a = tf.ones([3, 2], dtype=tf.int32)  # 变量的维度可变（不常用），但类型不可变
a = tf.fill([2, 3], 9)  # 产生一个指定大小的，给定数字的矩阵
a = tf.constant([1, 2, 3])  # 产生一个给定值的常量,，可为其指定shape
'''
# tf.Variable()函数用来声明一个变量，并为其赋初值
v1 = tf.Variable(tf.constant([[4.0, 3.0], [2.0, 1.0]]))
v2 = tf.Variable(tf.constant([[5.0, 6.0], [7.0, 8.0]]))
v1_init_value = v1.initialized_value()  # 获取v1变量的初始值
sess = tf.Session() # 会话拥有并管理TensorFlow程序运行时的所有资源
# sess=tf.InteractiveSession() #自动将生成的会话注册为默认会话，此后的eval()
# 和run()函数不需要指定是哪个session了
'''
with可以代替try…catch语句处理上下文环境产生的异常，
并且可以把with后面函数的返回值赋给as后面的变量。
该用法建立在上下文管理器之上。
'''
with sess.as_default():  # 将sess设为默认会话
    result = v1 * v2
    '''
    在TensorFlow中一个变量给出初始化声明后并没有执行，需要显式的调用初始化方法。
    可以用 sess.run(w.initializer) 的方式初始化单个变量，也可以用以下方式初始化
    所有变量。
    '''
    tf.global_variables_initializer().run()
    #sess.run(tf.global_variables_initializer()) #这种写法也可
    print("v1 * v2:\n", result.eval())  # 矩阵对应元素相乘
    print("v1 * v2:\n", sess.run(result))  # 具有相同效果
    print("v1 * v2:\n", result.eval(session=sess))  # 具有相同效果
    print("tf.matmul(v1,v2):\n", tf.matmul(v1, v2).eval())  # 矩阵乘法
    print("v1 + v2:\n", (v1 + v2).eval())  # 矩阵对应元素相加
    print("tf.add(v1,v2):\n", tf.add(v1, v2).eval())  # 矩阵加法，结果同上

    # tf.greater(v1,v2)是判断两个张量在元素级别的大小
    print("v1中每个元素是否比v2中的大：\n", tf.greater(v1, v2).eval())
    '''
    tf.where()有三个参数，第一个是判别条件，如果条件成立，则取值为第二个参数，反之为
    第三个，该函数是在作用在元素层面的。如果参数只有条件，则返回满足条件的元素下标。
    '''
    print("tf.where:\n", tf.where(tf.greater(v1, v2), v1, v2).eval())

'''
有选择的加载变量v1和v2，并为其重命名为v11和v22
saver=tf.train.Saver({"v11":v1,"v22":v2})

使用variables_to_restore函数，可以使在加载模型的时候将影子变量直接映射到变量的本身
ema = tf.train.ExponentialMovingAverage(0.99)  
ema.variables_to_restore()
'''
# 以下代码为模型的持久化实现
saver = tf.train.Saver()  # 该类用于保存模型
# 保存已经训练好的模型，包括计算图结构、参数取值等
saver.save(sess, "./model/model.ckpt")
'''
导出计算图的元图，并保存为json格式，TensorFlow中利用元图
来记录计算图中节点的信息以及运行计算图中节点所需要的元数据
'''
saver.export_meta_graph("./model/model.ckpt.meda.json", as_text=True)
# 以下两行代码为加载已经训练好的模型
# 第一行是加载已经持久化的图，避免了重复定义图上的运算
saver = tf.train.import_meta_graph("./model/model.ckpt.meta")
saver.restore(sess, "./model/model.ckpt")

'''
tf.ConfigProto()函数可以为会话配置参数
allow_soft_placement：为True时允许在满足某些条件的情况下，将GPU上运行的运算调整到CPU上
log_device_placement：为True时会在日志中记录每个节点所在的设备，以方便调试。为False时可减少日志量
'''
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)

# 以下为TensorFlow中常用的随机生成函数,可选参数有取值类型和随机数种子等
# 生成满足正态分布的一组数，参数为维度、均值（默认为0）和标准差
tf.random_normal([2, 3], mean=0, stddev=2)
# 截断正态分布，随机数会控制在距离均值两个标准差之内。参数有shape、均值和标准差
tf.truncated_normal([3, 5], mean=0, stddev=1, dtype="float")
# 均匀分布，其参数有shape，最小、最大值
tf.random_uniform([3, 3], minval=1, maxval=5)
# gamma分布，其参数有维度、形状参数alpha和尺度参数beta
tf.random_gamma([2,3],alpha=0.1, beta=0.3)

all_var = tf.global_variables()  # 获取当前计算图上所有的变量
trainable_var = tf.trainable_variables()  # 获取所有需要优化的参数

'''
当使用一个batch的样本进行训练时，定义多个变量是不现实的，可以用placeholder
机制来解决，该机制相当于定义了一个位置，这个位置的数据在程序运行时才指定。
这样就可以一次性计算多个样例的前向传播结果。
'''
w=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# 维度可以不定义，但是如果已知维度的话最好定义一下，可以降低错误率
x=tf.placeholder(tf.float32,shape=(3,2),name="input")
y=tf.matmul(x,w)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]})) #在运行时为输入x指定数据

global_step = tf.Variable(0)
print("global_step:\n", global_step)
'''
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
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
# learning_step=tf.train.GradientDescentOptimizer(learning_rate).minimize("loss function",global_step=global_step)

weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
    lambda_ = 0.5
    '''
    tf.contrib.layers.l1_regularizer()返回的是一个可以计算参数L1正则项的值的函数
    lambda_表示正则化项的权重，也就是total_loss = loss(theta) + lambda * R(w)中的
    lambda，theta是神经网络的参数
    '''
    print("L1 nomal:\n", sess.run(tf.contrib.layers.l1_regularizer(lambda_)(weights)))
    print("L2 nomal:\n", sess.run(tf.contrib.layers.l2_regularizer(lambda_)(weights)))

'''
tf.get_variable()函数用来创建一个新变量，或获取一个已存在的变量，该函数在使用时必须
指定变量名，其参数有变量名、维度和初始化等。初始化时和 tf.Variable()函数不同，可以用
的初始化函数有以下几种：
tf.constant_initializer
tf.random_normal_initializer
tf.truncated_normal_initailizer
tf.random_uniform_initializer
tf.uniform_unit_scaling_initializer
tf.zeros_initializer
tf.ones_initializer
'''
# tf.variabel_scope() # 用来生成一个上下文管理器
with tf.variable_scope("foo"):
    # 在命名空间"foo"中创建变量v，当该命名空间下存在同名变量时，则会报错
    v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))

# 将reuse设为True可在tf.get_variable函数中直接获取已声明的变量（不存在时会报错）
# tf.variable_scope()函数可以套用，当内层的reuse没指定时和外层保持一致
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])  # 直接获取已声明变量，当shape不为1时会报错
    print("v1:\n", v1)  # 输出为foo/v1:0，名称前是命名空间，命名空间可嵌套，后者是生成变量这个运算的第一个结果

v2 = tf.get_variable("foo/bar/v", [1])  # 可在创建时指定命名空间

# 可以通过以下代码来根据名字获取张量
# sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))


input = tf.get_variable("input", [64, 64,15,3], initializer=tf.random_normal_initializer(stddev=2))
# [5,5,3,16]表示的意思是卷积核矩阵的大小为5*5，输入层深度为3，输出层深度为16
filter_weight = tf.get_variable("weights", [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
biases = tf.get_variable("biases", [16], initializer=tf.constant_initializer(0.1))
'''
tf.nn.conv2d()来实现卷积层的前向传播算法，其参数分别为
input:输入图像，为四维矩阵，第一个维度为输入batch。input[0,:,:,:]
    表示第一张图片；后三个维度对应一个节点矩阵
filter_weight:卷积核矩阵
strides:四个维度的步长，但是第一维度和第四维度的值固定为1
padding:为SAME是表示全0填充，为VALID时表示不添加
'''
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding="SAME")
# 为每个节点都加上偏置项，这里不可以直接用加法
bias = tf.nn.bias_add(conv, biases)
active_conv = tf.nn.relu(bias)  # 使用relu函数作为激活函数
'''
tf.nn.max_pool()为最大池化，f.nn.avg_pool()为平均池化，其参数为
ksize：过滤器的尺寸，其中第一和第四维度的值固定为1
strides：步长
padding：填充方法
'''
pool = tf.nn.max_pool(active_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
########################################################################################################################################################################################
slim = tf.contrib.slim
net = slim.conv2d(input, 32, [3, 3])

# 读取图片原始数据
# tf.gfile.FastGFile()可以快速的获取文件句柄
image_raw_data = tf.gfile.FastGFile(r"C:\Users\zuzhiang\PycharmProjects\untitled2\Leslie.jpg", "rb").read()
with tf.Session() as sess:
    # 将图片解码为三维矩阵
    img_data = tf.image.decode_jpeg(image_raw_data)
    print("image:\n", img_data.eval())
    # 使用pyplot得到可视化图像
    plt.imshow(img_data.eval())
    plt.show()
    # 将图片数据转化为实数类型
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # 调整图片大小，method为调整算法，0是双线性插值法，1是最近邻法，
    # 2是双三次插值法，3是面积插值法
    resized = tf.image.resize_images(img_data, [600, 500], method=0)
    plt.imshow(resized.eval())
    plt.show()
    # 剪切或填充图片，当设定尺寸比原图小时则剪切，反之填充
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 300, 300)
    plt.imshow(croped.eval())
    plt.show()
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 700, 700)
    plt.imshow(padded.eval())
    plt.show()
    # 以原图中心为中心，按比例裁剪图片，第二个参数取值为(0,1]
    central_cropped = tf.image.central_crop(img_data, 0.7)
    plt.imshow(central_cropped.eval())
    plt.show()
    '''
    此外还有tf.image.crop_to_bounding_box()和tf.image.pad_to_bounding_box()
    函数来按照边框裁剪或填充图片，但是这两个函数都要求边框的大小和原图大小
    不能矛盾
    '''
    # 将图片上下翻转
    flipped = tf.image.flip_up_down(img_data)
    plt.imshow(flipped.eval())
    plt.show()
    # 将图片左右翻转
    flipped = tf.image.flip_left_right(img_data)
    plt.imshow(flipped.eval())
    plt.show()
    # 将图片沿对角线翻转
    transposed = tf.image.transpose_image(img_data)
    plt.imshow(transposed.eval())
    plt.show()
    '''
    #以50%的概率上下翻转图片
    tf.image.random_flip_up_down(img_data)
    #以50%的概率左右翻转图片
    tf.image.random_flip_left_right(img_data)
    '''
    # 将图片亮度-0.5
    adjusted = tf.image.adjust_brightness(img_data, -0.5)
    # 将图片的色彩截断在0~1之间
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()
    # 将图片亮度+0.5
    adjusted = tf.image.adjust_brightness(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()
    # 在[-max_delta,max_delta)范围内随机调整图片亮度
    adjusted = tf.image.random_brightness(img_data, max_delta=0.8)
    plt.imshow(adjusted.eval())
    plt.show()
    '''
    同样的
    tf.image.adjust_contrast(img_data,0.5) #调整对比度为原来的0.5倍
    tf.image.random_contrast(img_data,lower,upper) #在[lower,upper]范围内随机调整对比度
    tf.image.adjust_hue(img_data,0.9) #给原图色相加0.9
    tf.image.random_hue(img_data,max_delta) #max_delta取值在[0,0.5]之间
    tf.image.adjust_saturation(img_data,-5) #饱和度-5
    tf.image.random_saturation(img-data,lower,upper) #随机调整饱和度    
    '''
    # 将图片的三维矩阵中的数字均值变为0，方差变为1
    adjusted = tf.image.per_image_standardization(img_data)
    plt.imshow(adjusted.eval())
    plt.show()

    #tf.expand_dims(input,axis)用来沿着axis轴为input增加一维
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    # 标注框的四个参数为[Ymin,Xmin,Ymax,Xmax]，为原图像X、Y的百分比
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    '''
    tf.image.sample_distored_bounding_box()用来随机截取图片
    min_object_covered=0.4表示截取部分至少包含某个标注框的40%内容
    '''
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes, min_object_covered=0.4)
    # tf.image.draw_bounding_boxes()输入是一个batch数据，也就是多张图像组成的四维矩阵
    # 以下代码为图片加bounding box并绘制出来
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    plt.imshow(image_with_box[0].eval())
    plt.show()
    distorted_image = tf.slice(img_data, begin, size)
    plt.imshow(distorted_image.eval())
    plt.show()

    # 将三维矩阵重新编码，并存入文件中
    resized = tf.image.convert_image_dtype(resized, dtype=tf.uint8)
    encoded_image = tf.image.encode_jpeg(resized)
    with tf.gfile.GFile(r"C:\Users\zuzhiang\PycharmProjects\untitled2\Leslie_output.jpg", "wb") as f:
        f.write(encoded_image.eval())

    # 创建一个先进先出队列，最多可有2个元素，为整形
    q = tf.FIFOQueue(2, "int32")
    # 用q.enqueue_many()来初始化队列
    init = q.enqueue_many(([0, 10],))
    x = q.dequeue()  # 将队列中的第一个元素出队
    y = x + 1
    q_inc = q.enqueue([y])  # 入队

    with tf.Session() as sess:
        init.run()  # 进行初始化
        for _ in range(5):
            # 运行q_inc将执行数据出队、出队元素+1、重新入队整个过程
            v, _ = sess.run([x, q_inc])
            print("v:\n", v)

print("end")
