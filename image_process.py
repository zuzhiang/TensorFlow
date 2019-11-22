import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def distort_color(image,color_ordering=0):
    if color_ordering==0:
        image=tf.image.random_brightness(image,max_delta=32./255.)
        image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image=tf.image.random_hue(image,max_delta=0.2)
        image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image,0.0,1.0)

def preprocess_for_train(image,height,width,bbox):
    if bbox is None:
        bbox=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    if image.dtype != tf.float32:
        image=tf.image.convert_image_dtype(image,dtype=tf.float32)

    bbox_begin,bbox_size,_=tf.image.sample_distorted_bounding_box(
        tf.shape(image),bounding_boxes=bbox)
    distored_image=tf.slice(image,bbox_begin,bbox_size)
    distored_image=tf.image.resize_images(distored_image,[height,width],method=np.random.randint(4))
    distored_image=tf.image.random_flip_left_right(distored_image)
    distored_image=distort_color(distored_image,np.random.randint(2))
    return distored_image

if __name__=="__main__":
    image_raw_data=tf.gfile.FastGFile(r"C:\Users\zuzhiang\PycharmProjects\untitled2\Leslie.jpg","rb").read()
    sess=tf.InteractiveSession()
    img_data=tf.image.decode_jpeg(image_raw_data)
    boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    for i in range(6):
        result=preprocess_for_train(img_data,299,299,boxes)
        plt.imshow(result.eval())
        plt.show()