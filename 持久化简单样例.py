import tensorflow as tf
from tensorflow.python.framework import graph_util

v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2=tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
result=v1+v2

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

#导出当前图的GraphDef部分，只需该部分就可完成从输入层到输出层的计算过程
graph_def=tf.get_default_graph().as_graph_def()

#将图中的变量及其取值变为常量，同时将图中不必要的节点去掉
#最后一个参数需要给出计算节点的名称，所以只有add，没有后面的:0
output_graph_def=graph_util.convert_variables_to_constants(sess,graph_def,["add"])
model_filename=r"C:\Users\zuzhiang\PycharmProjects\untitled2\TensorFlow\model\combined_model.pb"

#将导出的模型存入文件
with tf.gfile.GFile(model_filename,"wb") as f:
    f.write(output_graph_def.SerializeToString())

#读取保存的模型文件
with tf.gfile.GFile(model_filename,"rb") as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())

    #将graph_def中保存的图加载到当前的图中
    #加载时给出的是张量的名称，故是add:0
    result=tf.import_graph_def(graph_def,return_elements=["add:0"])
    print("result:\n",result.eval())
