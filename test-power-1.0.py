# -*- coding:utf-8 -*-      
import cv2    
import tensorflow as tf    
import numpy as np 
from skimage import io,transform  
from sys import path

from skimage import io,transform
import os
import glob   

#读取图片及其标签函数
def read_image(path):
    images = []
    for img in glob.glob(path+'/*.tif'):
        print("reading the image:%s"%img)
        image = io.imread(img)
        image = transform.resize(image,(32,32,1))
        image = np.reshape(image , [-1,32,32,1])
        images.append(image)
    return np.asarray(images,dtype=np.float32)

def main():    
    sess = tf.InteractiveSession()    
#模型恢复  
    saver=tf.train.import_meta_graph('model_data/model.meta')  
   
    saver.restore(sess, 'model_data/model') 
    graph = tf.get_default_graph()  
      
    # 获取输入tensor,,获取输出tensor  
    x = sess.graph.get_tensor_by_name("x:0")  
    y_ = sess.graph.get_tensor_by_name("y_:0")  
    results = sess.graph.get_tensor_by_name("results:0") 
  
      

    # la = [2]
    # im = io.imread('./images/D.tif')

    # #调整大小    
    # im = transform.resize(im,(32,32,1)) 
    # im = np.reshape(im , [-1,32,32,1])

    # #类型转换
    # im = np.asarray(im,dtype=np.float32) 
    # la = np.asarray(la,dtype=np.int32)

    path = "./images/"
    switch = {
        '[0]': '0',
        '[1]': '1',
        '[2]': '2',
        '[3]': '3',
        '[4]': '4',
        '[5]': '5',
        '[6]': '6',
        '[7]': '7',
        '[8]': '8',
        '[9]': 'A',
        '[10]': 'B',
        '[11]': 'C',
        '[12]': 'D',
        '[13]': 'E',
        '[14]': 'F',
        '[15]': 'G',
        '[16]': 'H',
        '[17]': 'P',
        '[18]': 'S',
        '[19]': 'T',
        '[21]': 'Y',
    }
    result = []
    dataArr = read_image(path)
    print('单图测试：')
    for data in dataArr:
        res = sess.run(results, feed_dict={x:data,y_:[]})
        res = str(res)
        result.append(switch[res])
    print('识别结果为：', result) 
    #关闭会话    
    sess.close()  


      
    
if __name__ == '__main__':    
    main()  