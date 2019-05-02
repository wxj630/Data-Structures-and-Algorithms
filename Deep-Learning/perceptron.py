import numpy as np

# 定义sigmoid()函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 对模型参数进行初始化（权值w，偏置b）
def initilize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    #assert(w.shape == (dim, 1))
    #assert(isinstance(b, float) or isinstance(b, int))
    return w, b

# 前向传播函数，预测值y为模型从输入到经过激活函数处理后的输出的结果，损失函数采用交叉熵损失
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {'dw': dw,
             'db': db
             }

    return grads, cost

# 反向传播函数，计算每一步的当前损失，根据损失对权值进行更新
def backward_propagation(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    cost = []
    for i in range(num_iterations):
        grad, cost = propagate(w, b, X, Y)

        dw = grad['dw']
        db = grad['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            cost.append(cost)
        if print_cost and i % 100 == 0:
            print("cost after iteration %i: %f" % (i, cost))

    params = {"dw": w,
              "db": b
              }

    grads = {"dw": dw,
             "db": db
             }

    return params, grads, cost

# 一个简单的感知机就搭建起来了，定义一个预测函数对测试数据进行预测
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X)+b)
    for i in range(A.shape[1]):
        if A[:, i] > 0.5:
            Y_prediction[:, i] = 1
        else:
            Y_prediction[:, i] = 0

    assert(Y_prediction.shape == (1, m))
    return Y_prediction

# 对所有函数进行一下简单的封装
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initilize_with_zeros(X_train.shape[0])    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = backward_propagation(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d