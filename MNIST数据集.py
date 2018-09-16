import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
这一句需要从网络下载mnist数据集，往往无法实现,
下载到本地替代
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
'''
mnist = input_data.read_data_sets('./', one_hot=True)
tf.reset_default_graph()

#搭建模型
#None代表一次处理多少张图片
x = tf.placeholder(tf.float32,[None,784])#28*28
y = tf.placeholder(tf.float32,[None,10])#分成10组

#设置模型参数
W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

#正向传播
pred = tf.nn.softmax(tf.matmul(x,W)+b)

#反向传播 将生成的pred和样本标签y进行一次交叉熵运算，最小化误差cost
#mean求平均 sum求和
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

#参数设置
learning_rate = 0.1
#使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 50
batch_size = 100
display_step = 1
saver = tf.train.Saver()
model_path = "log/521model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size) #55000/100 每一轮训练多少数据
        print(total_batch)
        #遍历所有数据集
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #_是临时变量，使用它的原因可能是不需要第一个返回值
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            #计算平均值使得误差更平均
            avg_cost += c / total_batch

        if(epoch+1)%display_step == 0:
            print("Epoch:","%04d"%(epoch+1),"cost=","{:.9f}".format(avg_cost))

    print("训练完成")

    #测试model
    #预测数据和真实结果对比
    #arg max f(x): 当f(x)取最大值时，x的取值
    #arg min f(x)：当f(x)取最小值时，x的取值
    correct_prediction  =tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x: mnist.test.images,y: mnist.test.labels}))

    #存储模型
    save_path = saver.save(sess,model_path)
    print("model saved in file:%s"% save_path)
import pylab
#读取模型
print("Starting 2nd session")
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    #恢复模型并且读取所有变量参数进入test2
    saver.restore(sess2,model_path)
    #测试model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    output = tf.argmax(pred,1)
    batch_xs,batch_ys = mnist.train.next_batch(2)#返回两个手写体图片
    outputval,predv =sess2.run([output,pred],feed_dict={x:batch_xs})
    print(outputval,predv,batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
