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


def recognition():    
    sess = tf.InteractiveSession()    
    #模型恢复  
    saver=tf.train.import_meta_graph("./model_data/model.meta")  
   
    saver.restore(sess, "./model_data/model") 
    graph = tf.get_default_graph()  
      
    # 获取输入x,获取输出y_  
    x = sess.graph.get_tensor_by_name("x:0")  
    y_ = sess.graph.get_tensor_by_name("y_:0")  
    accuracy = sess.graph.get_tensor_by_name("accuracy:0") 

    #读取测试集
    test_data,test_label = read_image(test_path)
    acc = sess.run(accuracy, feed_dict={x:test_data,y_:test_label})
    print(acc/len(test_label))
    #关闭会话    
    sess.close()

recognition()