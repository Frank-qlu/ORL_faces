#author：Zhipeng Li
import numpy as np
import random,os,cv2,glob
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import hypertools as hyp
import time
import pywt
from sklearn.svm import  SVC
import matplotlib.pyplot as plt
batch_size = 40

def loadImageSet(folder=u'E:\data_faces', sampleCount=5): #加载图像集，随机选择sampleCount张图片用于训练
    trainData = []; testData = [];yTrain2=[];yTest2=[]
    #print(yTest)
    for k in range(40):
        yTrain1 = [k]
        yTest1 =[k]
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
                           return： 返回的值要注意，每一层的高频都是包含在一个tuple中，
                           例如三层的话返回为 [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2)， (cH1, cV1, cD1)]
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
        sample = random.sample(range(10), sampleCount)#random.sample()可以从指定的序列中，随机的截取指定长度的片断，不作原地修改
        #print(sample)
        #print(data)

        #sample=[1,3,5,7,9]
        #print(data_pwt)
        trainData.extend([data_pwt[i].ravel() for i in range(10) if i in sample])#ravel将多维数组降位一维####40*5*112*92
        testData.extend([data_pwt[i].ravel() for i in range(10) if i not in sample])#40*5*112*92

        #yTest.extend([]* (10-sampleCount))
        #yTrain.extend([k]* sampleCount)
        yTrain = np.matrix(yTrain1)
        yTrain= np.tile(yTrain1,5)#   np.tile(a,(2,1))就是把a先沿x轴（就这样称呼吧）复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍
        yTest=np.tile(yTest1,5)#沿着x轴复制5倍，增加列数
        yTrain2.extend(yTrain)
        yTest2.extend(yTest)
    return np.array(trainData),  np.array(yTrain2), np.array(testData), np.array(yTest2)

def main():
    #loadImageSet()
    xTrain_, yTrain, xTest_, yTest = loadImageSet()# 200*10304
    pca1=PCA(n_components=0.8)
    #hyp.plot(xTrain_,'o')
    xTrain_pca=pca1.fit_transform(xTrain_)# 把原始训练集映射到主成分组成的子空间中
    xTest_pca=pca1.transform(xTest_)# 把原始测试集映射到主成分组成的子空间中

    clf=SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
    clf.fit(xTrain_pca/255,yTrain)
    predict=clf.predict(xTest_pca / 255)
    print(clf.score(xTest_pca/255,yTest))
    print(u'支持向量机识别率: %.2f%%' % ((predict == np.array(yTest)).mean()*100)  )
if __name__ == '__main__':
    main()