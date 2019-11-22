import tensorflow as tf

reader=tf.TFRecorder()
filename_queue=tf.train.string_input_producer([r"C:\Users\zuzhiang\PycharmProjects\untitled2\TensorFlow\output.tfrecords"])

_,serialized_example=reader.redad(filename_queue)
features=tf.parse_single_example(
    serialized_example,
    features={
        "image_raw":tf.FixedLenFeature([],tf.string),
        "pixels":tf.FixedLenFeature([],tf.int64),
        "label":tf.FixedLenFeature([],tf.int64)})

image=tf.decode_raw(features["image_raw"],tf.uint8)
label=tf.cast(features["label",tf.int32])
pixels=tf.cast(features["pixels"],tf.int32)

sess=tf.Session()
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(10):
    print("image,label,pixels:\n",sess.run([image,label,pixels]))

