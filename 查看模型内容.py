
#查看w和b
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
savedir = "log/"
print_tensors_in_checkpoint_file(savedir+"linermodel.cpkt",None,True,True)

W = tf.Variable(1.0,name="weight")
b = tf.Variable(2.0,name="bias")

#颠倒放在一个字典
saver = tf.train.Saver({'weight':b,'bias':W})

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess,savedir+"linermodel.cpkt")

print_tensors_in_checkpoint_file(savedir+"linermodel.cpkt",None,True,True)

#保存检查点（checkpoint）的作用：训练可能持续很久，保持迭代的中间结果，避免中断之后需要重新训练