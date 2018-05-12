#coding=utf-8  
from __future__ import print_function  
  
# from tensorflow.examples.tutorials.mnist import input_data  
# mnist = input_data.read_data_sets("/tmp/data", one_hot=True)  
  
import tensorflow as tf  
from skimage import io,transform
import os
import glob
import numpy as np
# 定义网络超参数  
learning_rate = 0.001  
training_iters = 200000  
batch_size = 30  
display_step = 20  
  
# 定义网络参数  
n_input = 784 # 输入的维度  
n_classes = 21 # 标签的维度  
dropout = 0.5 # Dropout 的概率  

#训练图地址
train_path = "./icmt/test/"

#读取图片及其标签函数
def read_image(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    lables_0 = [0 for i in range(21)]
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.tif'):
            # print("reading the image:%s"%img)
            image = io.imread(img)
            image = transform.resize(image,(28,28,1))
            images.append(image)
            lables_0[index] = 0
            labels.append(lables_0)
    return np.asarray(images,dtype=np.float32), np.asarray(labels,dtype=np.float32)

#读取训练数据            
train_data,train_label = read_image(train_path)

#打乱训练数据
train_image_num = len(train_data)
train_image_index = np.arange(train_image_num)
np.random.shuffle(train_image_index)
train_data = train_data[train_image_index]
train_label = train_label[train_image_index]

  
# 占位符输入  
x = tf.placeholder(tf.float32, [None, 28, 28, 1])  
y = tf.placeholder(tf.float32, [None, 21])  
keep_prob = tf.placeholder(tf.float32)

#reshape
def use_reshape(input_shape):
    input_shape = input_shape.get_shape().as_list()
    nodes = input_shape[1]*input_shape[2]
    reshaped = tf.reshape(pool2,[-1,nodes])
    return reshaped  
  
# 卷积操作  
def conv2d(name, l_input, w, b):  
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)  
  
# 最大下采样操作  
def max_pool(name, l_input, k):  
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)  
  
# 归一化操作  
def norm(name, l_input, lsize=4):  
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)  
  
# 存储所有的网络参数  
weights = {  
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 64])),  
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),  
    'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),  
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),  
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),  
    'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),  
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),  
    'out': tf.Variable(tf.random_normal([4096, n_classes]))  
}  
biases = {  
    'bc1': tf.Variable(tf.random_normal([64])),  
    'bc2': tf.Variable(tf.random_normal([192])),  
    'bc3': tf.Variable(tf.random_normal([384])),  
    'bc4': tf.Variable(tf.random_normal([384])),  
    'bc5': tf.Variable(tf.random_normal([256])),  
    'bd1': tf.Variable(tf.random_normal([4096])),  
    'bd2': tf.Variable(tf.random_normal([4096])),  
    'out': tf.Variable(tf.random_normal([n_classes]))  
}  
  
# 定义整个网络  
def alex_net(_X, _weights, _biases, _dropout):  
    # 向量转为矩阵  
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])  
    
    # 第一层卷积  
    # 卷积  
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])  
    # 下采样  
    pool1 = max_pool('pool1', conv1, k=2)  
    # 归一化  
    norm1 = norm('norm1', pool1, lsize=4)  
  
    # 第二层卷积  
    # 卷积  
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])  
    # 下采样  
    pool2 = max_pool('pool2', conv2, k=2)  
    # 归一化  
    norm2 = norm('norm2', pool2, lsize=4)  
  
    # 第三层卷积  
    # 卷积  
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])  
    # 归一化  
    norm3 = norm('norm3', conv3, lsize=4)  
  
    # 第四层卷积  
    # 卷积  
    conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])  
    # 归一化  
    norm4 = norm('norm4', conv4, lsize=4)  
  
    # 第五层卷积  
    # 卷积  
    conv5 = conv2d('conv5', norm4, _weights['wc5'], _biases['bc5'])  
    # 下采样  
    pool5 = max_pool('pool5', conv5, k=2)  
    # 归一化  
    norm5 = norm('norm5', pool5, lsize=4)  
  
    # 全连接层1，先把特征图转为向量  
    dense1 = tf.reshape(norm5, [-1, _weights['wd1'].get_shape().as_list()[0]])  
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')  
    dense1 = tf.nn.dropout(dense1, _dropout)  
  
    # 全连接层2  
    dense2 = tf.reshape(dense1, [-1, _weights['wd2'].get_shape().as_list()[0]])  
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation  
    dense2 = tf.nn.dropout(dense2, _dropout)  
  
    # 网络输出层  
    out = tf.matmul(dense2, _weights['out']) + _biases['out']  
    return out  
  
# 构建模型  
pred = alex_net(x, weights, biases, keep_prob)  
  
# 定义损失函数和学习步骤  
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))  
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  
  
# 测试网络  
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
  
# 初始化所有的共享变量  
init = tf.initialize_all_variables()    


#每次获取batch_size个样本进行训练或测试
def get_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]

# 开启一个训练  
with tf.Session() as sess:  
    sess.run(init) 


    #将所有样本训练10次，每次训练中以64个为一组训练完所有样本。
    #train_num可以设置大一些。
    train_num = 6
    batch_size = 64

    for i in range(train_num):
        #训练
        train_loss,train_acc,batch_num = 0, 0, 0
        for train_data_batch,train_label_batch in get_batch(train_data,train_label,batch_size):
            # train_data_batch = use_reshape(train_data_batch)
            # train_label_batch = use_reshape(train_label_batch)
            _,err,acc = sess.run([optimizer,loss,accuracy],feed_dict={x:train_data_batch,y:train_label_batch,keep_prob: dropout})
            train_loss+=err;train_acc+=acc;batch_num += 1
        print(str(i) + ":")
        print("train loss:",train_loss/batch_num)
        print("train acc:",train_acc/batch_num)
        # log = str(i) + ":" + "\n" + "train loss:" + str(train_loss/batch_num) + "\n" + "train acc:" + str(train_acc/batch_num) + "\n\n"
        


    # step = 1  
    # # Keep training until reach max iterations  
    # while step * batch_size < training_iters:  
    #     batch_xs, batch_ys = mnist.train.next_batch(batch_size)  
    #     # 获取批数据  
    #     sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})  
    #     if step % display_step == 0:  
    #         # 计算精度  
    #         acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})  
    #         # 计算损失值  
    #         loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})  
    #         print ("Iter " + str(step*batch_size) + ", Minibatch Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))  
    #     step += 1  
    # print ("Optimization Finished!")  
    # # 计算测试精度  
    # print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))  

