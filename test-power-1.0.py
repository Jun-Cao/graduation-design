# -*- coding:utf-8 -*-      
import cv2    
import tensorflow as tf    
import numpy as np 
from skimage import io,transform  
from sys import path    
#用于将自定义输入图片反转  
# def reversePic(src):  
#         # 图像反转    
#     for i in range(src.shape[0]):  
#         for j in range(src.shape[1]):  
#             src[i,j] = 255 - src[i,j]  
#     return src   
            
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
  
      

    la = [2]
    im = io.imread('./images/8.tif')
    #调整大小    
    im = transform.resize(im,(32,32,1)) 
    im = np.reshape(im , [-1,32,32,1])  
    #类型转换
    im = np.asarray(im,dtype=np.float32) 
    la = np.asarray(la,dtype=np.int32)

    output = sess.run(results, feed_dict={x:im,y_:la})
    print('单图测试')
    print('y值为：', output)
    #关闭会话    
    sess.close()    
    
if __name__ == '__main__':    
    main()  