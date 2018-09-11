import tensorflow as tf
'''
#Variable变量（包括同一个变量）如果被赋予同一个名字，自动修改

var1 = tf.Variable(2.0,name="firstvar")
#var1: firstvar:0
print("var1:",var1.name)
var1 = tf.Variable(2.0,name="firstvar")
#var1: firstvar_1:0
print("var1:",var1.name)
var2 = tf.Variable(2.0,name="firstvar")
#var2: firstvar_2:0
print("var2:",var2.name)
#get_variable变量如果遇到同一个名字，报错

get_var1 = tf.get_variable("firstvar",[1],initializer=tf.constant_initializer)
print("get_var1:",get_var1.name)
get_var1 = tf.get_variable("firstvar_1",[1],initializer=tf.constant_initializer)
print("get_var1:",get_var1.name)
'''


##作用域

with tf.variable_scope("test1",):
    #shape = [2]一行两列
    var1 = tf.get_variable("firstvar", shape=[2],dtype=tf.float32)
    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar", shape=[2],dtype=tf.float32)

print("var1",var1.name)
print("var2",var2.name)

#要重用的话，布局必须和上一次相同，我尝试增加一个变量，结果报错
#重用意味着 var3 var4 就是var1 var2的别名？
with tf.variable_scope("test1",reuse=True):
    var3 = tf.get_variable("firstvar", shape=[2],dtype=tf.float32)
    with tf.variable_scope("test2"):
        var4 = tf.get_variable("firstvar", shape=[2],dtype=tf.float32)

print("var3",var3.name)
print("var4",var4.name)