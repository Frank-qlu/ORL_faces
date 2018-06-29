# ORL_faces
ORL人脸识别不同算法的实现，用到了scikit-learn,tensorflow等，任选5张训练，5张测试。因为每次训练随机挑选，所以每次输出识别率有偏差
算法                 识别率
bp神经网络             0.8
pca+bp神经网络         0.85
小波变换+pca+bp神经网络 0.95
CNN                    0.98
小波变换+pca+SVM        0.98
