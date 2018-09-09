#线性回归拟合二维数据示例代码
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#？
plotdata = {"batchsize":[],"loss":[]}

##随机生成模拟数据
train_X = np.linspace(-1,1,100)#生成-1到1的数据点
train_Y = 2*train_X + np.random.randn(100)*0.3
#'ro'点是红色的 ‘go’点是绿色的
plt.plot(train_X,train_Y,'go',label='Original data')
plt.legend()
plt.show()

##搭建模型
#占位符的作用,输入数据
X = tf.placeholder("float")
Y = tf.placeholder("float")
#模型参数
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")

#前向结构
# x * w + b 得到z 和y比较
z = tf.multiply(X,W)+b
tf.summary.histogram('z',z)#将预测值以直方图显示

#反向优化
cost = tf.reduce_mean(tf.square(Y-z))
tf.summary.scalar('loss_function',cost)#将损失用标量表示
#学习率
learning_rate = 0.01
#使用TF的梯度下降优化器 优化w和b使得cost最小，最终使得z和y的误差最小
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

##迭代训练模型
#初始化变量
init = tf.global_variables_initializer()
#训练参数，训练轮数 几轮打印一次
training_epochs = 20
display_step = 2
#保存模型
#保存最后一次检查点
saver =tf.train.Saver(max_to_keep=1)
savedir = "log/"
#启动session
with tf.Session() as sess:
    sess.run(init)
    merged_summary_op = tf.summary.merge_all()#合并所以的summary
    #创建summary_writer,用于写文件
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)

    #Fit all training data
    for epoch in range(training_epochs):
        for(x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            #生成summary
            summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y})
            summary_writer.add_summary(summary_str,epoch)


        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not(loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print("Finished")
    #保存了步长
    saver.save(sess,savedir+"linermodel.cpkt",global_step=epoch)

    #图形显示
    #我猜测w,b是优化器的代码修改，自己不能直观看到变化，很不舒服
    plt.plot(train_X,train_Y,'ro',label="Original data")
    plt.plot(train_X,sess.run(W)*train_X + sess.run(b),label="Fitted line")
    plt.legend()
    plt.show()

    ##使用模型
    print("x=0.2,z=",sess.run(z,feed_dict={X:0.2}))

#使用保存的模型
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,savedir+"linermodel.cpkt")
    print("2 x=0.2,z=",sess2.run(z,feed_dict={X:0.2}))
#w和x个数相同的原因？