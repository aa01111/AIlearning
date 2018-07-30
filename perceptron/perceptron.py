import numpy as np
from matplotlib import pyplot as plt

class Perceptron:
    total = 0
    #M1正实例点 M2负实例点
    M1 = np.array([[0,0]])
    M2 = np.array([[0,0]])
    M =[]

def get_train_data():
    train = []
    count = 0
    for line in open("train.txt", "r", encoding="utf-8"):
        train.append(list(map(eval,line.split())))
        count+=1
    Perceptron.total = count
    MA = np.array(train)
    MA = np.column_stack((MA, range(count)))

    for vector in MA:
        if      vector[2] == 1:
            if np.array_equal(Perceptron.M1,[[0,0]]):
                Perceptron.M1 = [vector[0:2]]
            else:
                Perceptron.M1 = np.append(Perceptron.M1,[vector[0:2]],axis=0)
        elif    vector[2] == -1:
            if np.array_equal(Perceptron.M2,[[0,0]]):
                Perceptron.M2 = [vector[0:2]]
            else:
                Perceptron.M2 = np.append(Perceptron.M2,[vector[0:2]],axis=0)



    Mtotal = np.vstack((Perceptron.M1,Perceptron.M2))
    min_x = np.min(Mtotal)
    max_x = np.max(Mtotal)

    # 分隔x,方便作图
    x = np.linspace(min_x, max_x, 100)
    return MA,x

def get_test_data():
    test = []
    count = 0
    for line in open("test.txt", "r", encoding="utf-8"):
        test.append(list(map(eval, line.split())))
        count += 1
    Perceptron.M = np.array(test)
    Perceptron.M = np.column_stack((Perceptron.M, range(count)))

    return Perceptron.M

# GRAM计算
def get_gram(MA):
    GRAM = np.empty(shape=(Perceptron.total,Perceptron.total))
    for i in range(len(MA)):
        for j in range(len(MA)):
            GRAM[i,j] = np.dot(MA[i,][:2], MA[j,][:2])
    return GRAM

def func(alpha,b,xi,yi,yN,index,GRAM):
    pa1 = alpha*yN
    pa2 = GRAM[:,index]
    num = yi*(np.dot(pa1,pa2)+b)
    return num

# 训练training data
def train(MA, alpha, b, GRAM, yN):
    error = 0
    for sample in MA:
        xi = sample[0:2]
        #读取列表中倒数第二个元素
        yi = sample[-2]
        #int 类型转化
        index = int(sample[-1])
        # 如果为误分类，改变alpha,b
        # n 为学习率 学习率等价于步长
        if func(alpha,b,xi,yi,yN,index,GRAM) <= 0:
            alpha[index] += n
            b += n*yi
            print(alpha,b)
            #迭代一次后，test
            iteration(alpha,b)
            error += 1;
    if error > 0:
        alpha,b = train(MA,  alpha, b, GRAM, yN)
    return alpha,b

# 作出分类线的图
def plot_classify(w,b,x, rate0):
    #还是没有懂
    y = (w[0]*x+b)/((-1)*w[1])
    #打印线
    plt.plot(x,y)
    plt.title('Accuracy = '+str(rate0))



def classify(w,b,test_i):
    #test实际是使用sign函数的过程
    #sign(超平面) wx+b>0 取1 wx+b<0取-1  wx+b=0 取0
    if np.sign(np.dot(w, test_i) + b) == 1:
        return 1
    else:
        return -1

# 测试数据，返回正确率
# 课本上同样没有讲test实现的过程

def test(w,b,test_data):
    count = 0
    right_count = 0
    for test_i in test_data[:,0:2]:
        classx = classify(w,b,test_i)
        if classx * test_data[count,2] == 1:
            right_count += 1
        count += 1
    rate  = right_count/len(test_data)
    return rate

def iteration(alpha,b):
    alphap = np.column_stack((alpha * yN, alpha * yN))
    w = sum(alphap * xN)
    rate = test(w, b, test_data)
    plt.plot(Perceptron.M1[:, 0], Perceptron.M1[:, 1], 'ro')
    plt.plot(Perceptron.M2[:, 0], Perceptron.M2[:, 1], 'go')
    plt.plot(Perceptron.M[:, 0], Perceptron.M[:, 1], "*y")
    plot_classify(w, b, x, rate)
    plt.show()

if __name__=="__main__":


    MA,x= get_train_data()
    test_data = get_test_data()
    GRAM = get_gram(MA)

    yN = MA[:,2]
    xN = MA[:,0:2]
    alpha = [0]*Perceptron.total
    b = 0
    n = 1
    alpha, b = train(MA, alpha, b, GRAM, yN)

