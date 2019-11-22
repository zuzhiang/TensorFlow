import tensorflow as tf

v1=tf.Variable(0,dtype=tf.float32)
step=tf.Variable(0,trainable=False)

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
ema=tf.train.ExponentialMovingAverage(0.99,step)
print("ema:\n",ema)
# 定义一个更新变量滑动平均的操作，这里需要给定一个列表（[v1]），每次执行
# 该操作时，这个列表中的变量都会被更新
maintain_averages_op=ema.apply([v1])

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    # 通过ema.average(v1)来获取滑动平均之后变量的值，在初始化之后
    # 变量v1的值和v1的滑动平均都为0
    print(sess.run([v1,ema.average(v1)]))

    # 更新v1的值为5
    sess.run(tf.assign(v1,5))
    # 更新v1的滑动平均值，衰减率为 min{0.99,(1+step)/(10+step)}=0.1
    # 所以v1的滑动平均会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))

    #tf.assign(a,b)的作用是将b的值赋给a
    sess.run(tf.assign(step,10000))
    sess.run(tf.assign(v1,10))
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))

    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))
