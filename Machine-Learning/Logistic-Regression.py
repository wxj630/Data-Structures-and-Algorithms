import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

class logistic_regression():
    def __init__(self):
        pass

    # 定义一个sigmoid函数
    def sigmoid(self,x):
        z = 1 / (1 + np.exp(-x))
        return z


    # 模型参数初始化
    def initialize_params(self,dims):
        W = np.zeros((dims, 1))
        b = 0
        return W, b


    # 逻辑回归模型主体
    def logistic(self,X, y, W, b):
        num_train = X.shape[0]
        num_fearure = X.shape[1]

        a = sigmoid(np.dot(X,W) + b) # sigema(z)
        cost = -1/num_train * np.sum(y*np.log(a) + (1-y)*np.log(1-a)) # loss function

        dW = np.dot(X.T,(a-y)/num_train) # 求偏导
        db = np.sum(a-y)/num_train

        return a,cost,dW,db

    # 定义基于梯度下降的参数更新训练过程
    def logistic_train(self,X,y,learning_rate,epochs):
        # 初始化模型参数
        W,b = initialize_params(X.shape[1])
        cost_list = []

        # 迭代训练
        for i in range(epochs):
            # 计算当前次的模型计算结果、损失核参数梯度
            a,cost,dW,db = logistic(X,y,W,b)
            # 参数更新
            W = W - learning_rate * dW
            b = b - learning_rate * db

            # 记录损失
            if i % 100 == 0:
                cost_list.append(cost)
            # 打印训练过程中的损失
            if i % 100 == 0:
                print('epoch %d cost %f' % (i,cost))

        # 保存参数
        params = {
            'W':W,
            'b':b
        }

        # 保存梯度
        grads = {
            'dW': dW,
            'db': db
        }

        return cost_list,params,grads

    # 定义对测试数据的预测函数
    def predict(self,X,params):
        y_prediction = sigmoid(np.dot(X,params['W']) + params['b'])
        for i in range(len(y_prediction)):
            if y_prediction[i] > 0.5:
                y_prediction[i] = 1
            else:
                y_prediction[i] = 0
        return y_prediction

    # 定义一个分类准确率函数来评估
    def accuracy(self,y_test, y_pred):
        correct_count = 0
        for i in range(len(y_test)):
            for j in range(len(y_pred)):
                if y_test[i] == y_pred[j] and i == j:
                    correct_count +=1

        accuracy_score = correct_count / len(y_test)
        return accuracy_score

    # 使用sklearn模拟生成的二分类数据集进行模型训练和测试
    def create_data(self):
        X, labels = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, random_state=1,
                                        n_clusters_per_class=2)
        labels = labels.reshape((-1, 1))
        offset = int(X.shape[0] * 0.9)
        X_train, y_train = X[:offset], labels[:offset]
        X_test, y_test = X[offset:], labels[offset:]
        return X_train, y_train, X_test, y_test

    # 定义一个绘制模型决策边界的图形函数
    def plot_logistic(self,X_train, y_train, params):
        n = X_train.shape[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if y_train[i] == 1:
                xcord1.append(X_train[i][0])
                ycord1.append(X_train[i][1])
            else:
                xcord2.append(X_train[i][0])
                ycord2.append(X_train[i][1])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=32, c='red')
        ax.scatter(xcord2, ycord2, s=32, c='green')
        x = np.arange(-1.5, 3, 0.1)
        y = (-params['b'] - params['W'][0] * x) / params['W'][1]
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
    plt.show()

if __name__== '__main__':
    model = logistic_regression()
    X_train, y_train, X_test, y_test = model.create_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    cost_list, params, grads = model.logistic_train(X_train, y_train, 0.01, 1000)
    print(params)
    y_train_pred = model.predict(X_train, params)
    accuracy_score_train = model.accuracy(y_train, y_train_pred)
    print('train accuracy is:', accuracy_score_train)
    y_test_pred = model.predict(X_test, params)
    accuracy_score_test = model.accuracy(y_test, y_test_pred)
    print('test accuracy is:', accuracy_score_test)
    model.plot_logistic(X_train, y_train, params)



