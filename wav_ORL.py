#author：(李志鹏)Frank Lee
#2019 年3 月22日
import numpy as np
import tensorflow as tf

import random,os,cv2,glob
from sklearn.decomposition import PCA
#import hypertools as hyp
import time
import pywt
import matplotlib.pyplot as plt
batch_size = 40

def loadImageSet(folder=u'E:\data_faces', sampleCount=5): #加载图像集，随机选择sampleCount张图片用于训练
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
        data_pwt=[]
        """           
                           ###多尺度二维离散小波分解wavedec2
                           ###c为各层分解系数，s为各层分解系数长度
                           pywt.wavedec2(data, wavelet, mode=’symmetric’, level=None, axes=(-2, -1))
                           data: 输入的数据
                           wavelet:小波基
                           level: 尺度（要变换多少层）
                           return： 返回的值要注意，每一层的高频都是包含在一个tuple中，例如三层的话返回为 [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2)， (cH1, cV1, cD1)]
        ###保留低频部分，（高频部分为噪声部分）
        """
        for i in range(10):
            #subplot(numRows, numCols, plotNum)图表的整个绘图区域被分成 numRows 行和 numCols 列,然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
            #plt.subplot(1,2,1)
            #plt.imshow(data[0],cmap="gray")

            xTrain_pwt = pywt.wavedec2(data[i], 'db2', mode='symmetric', level=3, axes=(-2, -1))
            #plt.subplot(1, 2, 2)
            #plt.imshow(xTrain_pwt[0],cmap="gray")
            #plt.show()
            data_pwt.append(xTrain_pwt[0])

        #print(xTrain_pwt)

        sampleCount=5
        #sample = random.sample(range(10), sampleCount)#random.sample()可以从指定的序列中，随机的截取指定长度的片断，不作原地修改
        #print(sample)
        #print(data)

        sample=[1,3,5,7,9]
        #print(data_pwt)
        trainData.extend([data_pwt[i].ravel() for i in range(10) if i in sample])#ravel将多维数组降位一维####40*5*112*92
        testData.extend([data_pwt[i].ravel() for i in range(10) if i not in sample])#40*5*112*92
        yTrain1[k]=1
        yTest1[k]=1
        #yTest.extend([]* (10-sampleCount))
        #yTrain.extend([k]* sampleCount)
        yTrain = np.matrix(yTrain1)
        yTrain= np.tile(yTrain1,5)#   np.tile(a,(2,1))就是把a先沿x轴（就这样称呼吧）复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍
        yTest=np.tile(yTest1,5)#沿着x轴复制5倍，增加列数
        yTrain2.extend(yTrain)
        yTest2.extend(yTest)
    return np.array(trainData),  np.array(yTrain2), np.array(testData), np.array(yTest2)
#构造添加一个神经层的函数
def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    layer_name="layer%s"%n_layer
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b')
            tf.summary.histogram(layer_name+"/biases",biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs',outputs)
        return outputs
def main():
    #loadImageSet()
    xTrain_, yTrain, xTest_, yTest = loadImageSet()# 200*10304
    yTrain.shape=200,40
    yTest.shape=200,40
    pca1=PCA(n_components=0.80)
    #hyp.plot(xTrain_,'o')
    xTrain_pca=pca1.fit_transform(xTrain_)# 把原始训练集映射到主成分组成的子空间中
    xTest_pca=pca1.transform(xTest_)# 把原始测试集映射到主成分组成的子空间中
    #hyp.plot(xTrain_pca,'o')
    # 放置占位符，用于在计算时接收输入值
    with tf.name_scope('inputs'):
        # define placeholder for inputs to network
        x = tf.placeholder("float", [None,15],name='x_in')
        y_ = tf.placeholder("float", [None, 40],name='y_in')
    #定义隐含层，输入神经元个数=特征10304
    l1 = add_layer(x,15,50,n_layer=1 ,activation_function=tf.nn.sigmoid)
    #定义输出层。此时的输入就是隐藏层的输出——90，输入有90层（隐藏层的输出层），输出有40层
    y= add_layer(l1,50,40,n_layer=2, activation_function=tf.nn.softmax)

    # 计算交叉墒
    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        #cross_entropy=tf.reduce_mean(tf.square(y_-y))
        tf.summary.scalar('loss', cross_entropy )  # 用于观察常量的变化

    # 定义命名空间，使用tensorboard进行可视化
    with tf.name_scope('train'):
        # 使用梯度下降算法以0.01的学习率最小化交叉墒
        train_step = tf.train.AdagradOptimizer(1).minimize(cross_entropy)
    # 开始训练模型，循环1000次，每次都会随机抓取训练数据中的10条数据，然后作为参数替换之前的占位符来运行train_step
    def return_next_batch(batch_size, pos):
        start = pos * batch_size
        end = start + batch_size
        return xTrain_pca[start:end], yTrain[start:end]

    # 启动初始化，为其分配固定数量的显存
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #tf.merge_all_summaries() 方法会对我们所有的 summaries 合并到一起
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', tf.get_default_graph())
    # 初始化之前创建的变量的操作
    sess.run(tf.global_variables_initializer())
    beginTime=time.time()
    for i in range(10000):
        for j in range(5):
            batch_x, batch_y =return_next_batch(batch_size, j)
            batch_x=batch_x/255    #归一化
            # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
            sess.run(train_step, feed_dict={x: np.matrix( batch_x), y_: np.matrix(batch_y)})
            #print(sess.run(cross_entropy, feed_dict={x: np.matrix((batch_x)), y_: np.matrix(batch_y)}))
            if(sess.run(cross_entropy, feed_dict={x: np.matrix((batch_x)), y_: np.matrix(batch_y)})==0):
                break
        if i % 50 == 0:  # 每训练50次，合并一下结果
            result = sess.run(merged, feed_dict={x: np.matrix((batch_x)), y_: np.matrix(batch_y)})
            writer.add_summary(result, i)

    endTime = time.time()
    costTime=endTime-beginTime
    print("训练时间:%f" %(costTime))
    # 评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 计算正确预测项的比例，因为tf.equal返回的是布尔值，使用tf.cast可以把布尔值转换成浮点数，tf.reduce_mean是求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 在session中启动accuracy，输入是orl中的测试集
    print("识别率:%.2f%%" %((sess.run(accuracy, feed_dict={x:np.matrix(xTest_pca/255), y_: np.matrix(yTest)}))*100))

if __name__ == '__main__':
    main()