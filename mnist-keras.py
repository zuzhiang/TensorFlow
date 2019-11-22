import keras
import cv2 as cv
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

num_classes = 10
img_rows, img_cols = 28, 28

# trainX是图像，共6w张28*28大小的图像；trainY是图像的真实分类
(trainX, trainY), (testX, testY) = mnist.load_data()
print("trainX.shape: ",trainX.shape)
print("trainX[0]:\n",trainX[0])
# 根据图片编码方式来设置输入层格式
if K.image_data_format() == "channels_first":
    trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
    testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols) #第一个参数为通道数，因是黑白图，故为1
else:
    trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
    testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#将每个像素值转化为0~1之间的实数
trainX = trainX.astype("float32")
testX = testX.astype("float32")
trainX /= 255.0
testX /= 255.0
#把真实分类转化为one-hot编码
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)

model = Sequential() #声明一个顺序模型，也可把各层按顺序放到Sequential的参数中
#第一层为深度为32的卷积层，必须指定输入尺寸
model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2))) #最大池化层
model.add(Conv2D(64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) #将卷积层的输出展开为一维数组后作为全连接层的输入
model.add(Dense(500, activation="relu")) #全连接层，有500个节点
model.add(Dense(num_classes, activation="softmax")) #全连接层，有num_classes个节点
#定义损失函数、优化方法和测评方法
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"])
#指定训练集数据、batch大小、训练轮数和验证集数据，keras可以自动完成模型训练过程
model.fit(trainX, trainY, batch_size=128, epochs=20, validation_data=(testX, testY))
score = model.evaluate(testX, testY) #在测试集上计算准确率
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
