import tensorflow as tf
#tf.nn.softmax_cross_entropy_with_logits等价于tf.nn.softmax+ -tf.reduce_sum(labels*tf.log(logits_scaled),1)
labels = [[0,0,1],[0,1,0]]

logits = [[2,0.5,6],[0.1,0,3]] # w*x+b = logits
logits_scaled = tf.nn.softmax(logits) #第一个向量归一化后的第一个元素e^2/(e^2+e^0.5+e^6)
logits_scaled2 = tf.nn.softmax(logits_scaled)#这句没有用 纯粹再归一化一次
#result1等价于result3
result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits_scaled)#这是错误操作
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)
a = tf.log(logits_scaled)
with tf.Session() as sess:

    print("scaled=",sess.run(logits_scaled))
    print(sess.run(a))
    print("scaled2=",sess.run(logits_scaled2))

    print("rel1=", sess.run(result1),"\n")
    print("rel2=", sess.run(result2),"\n")
    print("rel3=", sess.run(result3))

#标签总概率为1
labels = [[0.4,0.1,0.5],[0.3,0.6,0.1]]
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
with tf.Session() as sess:
    print("rel4=",sess.run(result4),"\n")

#sparse
labels = [2,1]
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
with tf.Session() as sess:
    print("rel5=",sess.run(result5),"\n")

loss = tf.reduce_mean(result1)
with tf.Session() as sess:
    print("loss=",sess.run(loss))

labels = [[0,0,1],[0,1,0]]
loss2 = -tf.reduce_sum(labels*tf.log(logits_scaled))

with tf.Session() as sess:
    print("loss2=",sess.run(loss2))