#模型保存以及单个图片的识别
from skimage import io,transform
import os
import glob
import numpy as np
import tensorflow as tf
import time


#将所有的图片重新设置尺寸为32*32
w = 32
h = 32
c = 1

#mnist数据集中训练数据和测试数据保存地址
test_path = "./icmt/test/"


#读取图片及其标签函数
def read_image(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.tif'):
            # print("reading the image:%s"%img)
            image = io.imread(img)
            image = transform.resize(image,(w,h,c))
            images.append(image)
            labels.append(index)
    return np.asarray(images,dtype=np.float32),np.asarray(labels,dtype=np.int32)

#读取训练数据及测试数据   
test_data,test_label = read_image(test_path)

#每次获取batch_size个样本进行训练或测试
def get_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]         



sess = tf.InteractiveSession()    
#模型恢复  
saver=tf.train.import_meta_graph("./model_data/model.meta")  

saver.restore(sess, "./model_data/model") 
graph = tf.get_default_graph()  
  
# 获取输入x,获取输出y_  
x = sess.graph.get_tensor_by_name("x:0")  
y_ = sess.graph.get_tensor_by_name("y_:0")  
dropout_rate = sess.graph.get_tensor_by_name("do_rate:0") 
results = sess.graph.get_tensor_by_name("results:0") 
accuracy = sess.graph.get_tensor_by_name("accuracy:0") 


test_num = 1
batch_size = 64

#
for i in range(test_num):
    #训练
    test_loss,test_acc,batch_num = 0, 0, 0
    test_res = []
    for test_data_batch,test_label_batch in get_batch(test_data,test_label,batch_size):
        res,acc = sess.run([results,accuracy],feed_dict={x:test_data_batch, y_:test_label_batch, dropout_rate:1.0})
        test_acc += acc;batch_num += 1;test_res.append(res)
    print("准确率")
    print("train acc:",test_acc/batch_num)
    # print("测试结果：")
    # print(test_res)
# 

#关闭会话    
sess.close()
