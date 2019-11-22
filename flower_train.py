#以下代码可将图片数据分为训练、验证和测试三个数据集，并且
#可以把jpg格式转化为inception-v3可处理的299*299*3的矩阵
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

INPUT_DATA=r"C:\Users\zuzhiang\PycharmProjects\untitled2\TensorFlow\flower_photos"
OUTPUT_FILE=r"C:\Users\zuzhiang\PycharmProjects\untitled2\TensorFlow\flower_photos\flower_processed_data.npy"
#测试数据集和验证数据集的比例
VALIDATION_PERCENTAGFE=10
TEST_PERCENTAGE=10

#读取数据并将其分割成训练数据、验证数据和测试数据
def create_image_lists(sess,testing_percentage,validation_percentage):
    #os.walk()可实现目录遍历，通过在目录树中游走输出在目录中的文件名
    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]
    print("sub_dir:\n",sub_dirs)
    is_root_dir=True

    #初始化各个数据集
    train_images=[]
    train_labels=[]
    testing_images=[]
    testing_labels=[]
    validation_images=[]
    validation_labels=[]
    current_label=0

    #读取所有的子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue
        #获取一个子目录中所有的图片
        extensions=["jpg","jpeg","JPG","JPEG"]
        file_list=[]
        #os.path.basename()返回path最后的文件名
        dir_name=os.path.basename(sub_dir)
        print("dir_name:\n",dir_name)
        for extension in extensions:
            file_glob=os.path.join(INPUT_DATA,dir_name,"*."+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:continue
        print("file_list:\n",file_list)

        # 处理图片数据
        for file_name in file_list:
            #将图片转化为299*299
            image_raw_data=gfile.FastGFile(file_name,"rb").read()
            image=tf.image.decode_jpeg(image_raw_data)
            if image.dtype!=tf.float32:
                image=tf.image.convert_image_dtype(image,dtype=tf.float32)
            image=tf.image.resize_images(image,[299,299])
            image_value=sess.run(image)

            #随机划分数据集
            chance=np.random.randint(100)
            if chance<validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance<(testing_percentage+validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                train_images.append(image_value)
                train_labels.append(current_label)
        current_label+=1
    #将数据集打乱以获得更好的训练效果
    #get_state()：可理解为设定状态，记录下数组被打乱的操作
    state=np.random.get_state()
    #打乱train_images
    np.random.shuffle(train_images)
    #set_state()：接收get_state()返回的值，并进行同样的操作
    np.random.set_state(state)
    #以相同的顺序打乱train)labels
    np.random.shuffle(train_labels)
    #np.asanyarray()与np.array()类似，但array对目标做一个拷贝，而asarray不会
    return np.asanyarray([train_images,train_labels,validation_images,validation_labels,testing_images,testing_images])

def main():
    with tf.Session() as sess:
        processed_data=create_image_lists(sess,TEST_PERCENTAGE,VALIDATION_PERCENTAGFE)
        #通过numpy格式保存处理后的数据
        np.save(OUTPUT_FILE,processed_data)

if __name__=="__main__":
    main()
    print("end")