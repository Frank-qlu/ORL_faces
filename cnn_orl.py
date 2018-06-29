'''
author：zhipeng li
'''
import numpy as np
import tensorflow as tf
import random,os,cv2,glob
import time

batch_size = 50


def loadImageSet(folder=u'data_faces', sampleCount=5): #加载图像集，随机选择sampleCount张图片用于训练
    trainData = []; testData = [];yTrain2=[];yTest2=[]
    #print(yTest)
    for k in range(40):
        yTrain1 = np.zeros(40)
        yTest1 = np.zeros(40)
        folder2 = os.path.join(folder, 's%d' % (k+1))
        """
        a=glob.glob(os.path.join(folder2, '*.pgm'))
        for d in a:
            #print(d)
            img=cv2.imread(d)#imread读取图像返回的是三维数组,返回值是3个数组：I( : , : ,1) I( : , : ,2) I( : , : ,3) 这3个数组中应该存放的是RGB的值
            #print(img)#112*92*3
            cv2.imshow('image', img)
        """
        #data 每次10*112*92
        data = [ cv2.imread(d,0) for d in glob.glob(os.path.join(folder2, '*.pgm'))]#cv2.imread()第二个参数为0的时候读入为灰度图，即使原图是彩色图也会转成灰度图#glob.glob匹配所有的符合条件的文件，并将其以list的形式返回
        sample = random.sample(range(10), 5)#random.sample()可以从指定的序列中，随机的截取指定长度的片断，不作原地修改
        #print(sample)
        #sample=[0,1,2,3,4]
        #print(sample)
        trainData.extend([data[i].ravel() for i in range(10) if i in sample])#ravel将多维数组降位一维####40*5*112*92
        testData.extend([data[i].ravel() for i in range(10) if i not in sample])#40*5*112*92
        yTrain1[k]=1
        yTest1[k]=1
        #yTest.extend([]* (10-sampleCount))
        #yTrain.extend([k]* sampleCount)
        yTrain = np.matrix(yTrain1)
        yTrain= np.tile(yTrain1,5)#   np.tile(a,(2,1))就是把a先沿x轴（就这样称呼吧）复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍
        yTest=np.tile(yTest1,5)#沿着x轴复制5倍，增加列数
        """
        yTrain.shape=5,40
        yTest.shape=5,40
       """
        yTrain2.extend(yTrain)
        yTest2.extend(yTest)
    return np.array(trainData),  np.array(yTrain2), np.array(testData), np.array(yTest2)

def weight_variable(shape):
    inital=tf.truncated_normal(shape,stddev=0.1) #产生随机变量tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定
    return tf.Variable(inital)
def bias_variable(shape):
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital)
def conv2d(x,W):
    #stride[1,x_movment,y_movment,1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def main():
    #loadImageSet()
    xTrain_, yTrain, xTest_, yTest = loadImageSet()# 200*10304
    yTrain.shape=200,40
    yTest.shape=200,40

    def return_next_batch(batch_size, pos):
        start = pos * batch_size
        end = start + batch_size
        return xTrain_[start:end], yTrain[start:end]
    ###定义变量占位符
    x = tf.placeholder(tf.float32, [None, 10304])#112X92
    y_=tf.placeholder(tf.float32,[None,40])
    keep_prob=tf.placeholder(tf.float32)
    x_image=tf.reshape(x,[-1,112,92,1])# -1表示任意数量的样本数，chanel黑白为1
    print(x_image.shape)


    ####conv1 layer###
    W_conv1=weight_variable([32,32,1,32])#patch 32x32,in size=1,0utsize=32
    b_conv1=bias_variable([32])
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output size 112x92x32
    h_pool1=max_pool_2x2(h_conv1) #output size 56x46x32

    #####conv2 layer###
    W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5,in size=32,0utsize=64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 56x46x64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 28x23x64
    print(h_conv2)

    ##func1 layer##
    W_fc1=weight_variable([28*23*64,800])
    b_fc1=bias_variable([800])
    # [n_samples, 28, 23, 64] ->> [n_samples, 28*23*64]
    h_pool2_fat=tf.reshape(h_pool2,[-1,28*23*64])
    h_fc1=tf.nn.sigmoid(tf.matmul(h_pool2_fat,W_fc1)+b_fc1)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)  ###dropout防止过拟合###

    ##func2 layer##
    W_fc2 = weight_variable([800, 40])
    b_fc2 = bias_variable([40])
    prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

    ###定义损失函数和优化器
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction)))
    train_step = tf.train.AdagradOptimizer(1e-3).minimize(cross_entropy)

    def compute_accuracy(v_xs, v_ys):
        y_pre = sess.run(prediction, feed_dict={x: v_xs, keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={x: v_xs, y_: v_ys, keep_prob: 1})
        return result

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        for j in range(4):
            batch_x, batch_y =return_next_batch(batch_size, j)
            batch_x=batch_x/255    #归一化
            # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
            sess.run(train_step, feed_dict={x: np.matrix( batch_x), y_: np.matrix(batch_y),keep_prob:0.5})
        if i%50==0:
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
            # 计算正确预测项的比例，因为tf.equal返回的是布尔值，使用tf.cast可以把布尔值转换成浮点数，tf.reduce_mean是求平均值
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # 在session中启动accuracy，输入是orl中的测试集
            print("识别率:%.2f%%" % ((sess.run(accuracy, feed_dict={x: np.matrix(xTest_ / 255), y_: np.matrix(yTest),
                                                                 keep_prob: 1})) * 100))
            # 计算正确预测项的比例，因为tf.equal返回的是布尔值，使用tf.cast可以把布尔值转换成浮点数，tf.reduce_mean是求平均值

            # 在session中启动accuracy，输入是orl中的测试集


            #print(sess.run(cross_entropy, feed_dict={x: np.matrix((batch_x)), y_: np.matrix(batch_y)}))
            #if(sess.run(cross_entropy, feed_dict={x: np.matrix((batch_x)), y_: np.matrix(batch_y)})==0):
               # break
    # 评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
    # 计算正确预测项的比例，因为tf.equal返回的是布尔值，使用tf.cast可以把布尔值转换成浮点数，tf.reduce_mean是求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 在session中启动accuracy，输入是orl中的测试集
    print("识别率:%.2f%%" %((sess.run(accuracy, feed_dict={x:np.matrix(xTest_/255), y_: np.matrix(yTest),keep_prob: 1}))*100))



if __name__ == '__main__':
    main()