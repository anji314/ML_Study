```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

data = np.loadtxt('./data.csv',delimiter=',',unpack=True, dtype = 'float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

global_step=tf.Variable(0,trainable=False,name='global_step')

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
```

    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    


```python
with tf.name_scope('layer1'):
    W1=tf.Variable(tf.random_uniform([2,10],-1.,1.),name='W1')
    L1=tf.nn.relu(tf.matmul(X,W1))
    
with tf.name_scope('layer2'):
    W2=tf.Variable(tf.random_uniform([10,20],-1.,1.),name='W2')
    L2=tf.nn.relu(tf.matmul(L1,W2))

with tf.name_scope('output'):
    W3=tf.Variable(tf.random_uniform([20,3],-1.,1.),name='W3')
    model=tf.nn.relu(tf.matmul(L2,W3))
    
with tf.name_scope('optimizer'):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=model))
    
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
    train_op=optimizer.minimize(cost,global_step=global_step)
```

    WARNING:tensorflow:From <ipython-input-2-0582a2d5ba7e>:14: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    


```python
tf.summary.scalar('cost',cost)
```




    <tf.Tensor 'cost:0' shape=() dtype=string>




```python
sess=tf.Session()
saver=tf.train.Saver(tf.global_variables())

ckpt=tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

```


```python
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('./logs',sess.graph)
```


```python
for step in range(100):
    sess.run(train_op,feed_dict={X:x_data,Y:y_data})
    
    print('Step : %d, '%sess.run(global_step),
         'Cost : %.3f'%sess.run(cost,feed_dict={X:x_data,Y:y_data}))
   
    summary=sess.run(merged,feed_dict={X:x_data,Y:y_data})
    writer.add_summary(summary,global_step=sess.run(global_step))
    

saver.save(sess,'./model/dnn.ckpt',global_step=global_step)
   
prediction=tf.argmax(model,1)
target=tf.argmax(Y,1)
print('예측값 : ',sess.run(prediction,feed_dict={X:x_data}))
print('실제값 : ',sess.run(target,feed_dict={Y:y_data}))

is_correct=tf.equal(prediction,target)
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도 : %.2f' %sess.run(accuracy*100,feed_dict={X:x_data,Y:y_data}))
```

    Step : 301,  Cost : 0.916
    Step : 302,  Cost : 0.916
    Step : 303,  Cost : 0.916
    Step : 304,  Cost : 0.916
    Step : 305,  Cost : 0.916
    Step : 306,  Cost : 0.916
    Step : 307,  Cost : 0.916
    Step : 308,  Cost : 0.916
    Step : 309,  Cost : 0.916
    Step : 310,  Cost : 0.916
    Step : 311,  Cost : 0.916
    Step : 312,  Cost : 0.916
    Step : 313,  Cost : 0.916
    Step : 314,  Cost : 0.916
    Step : 315,  Cost : 0.916
    Step : 316,  Cost : 0.916
    Step : 317,  Cost : 0.916
    Step : 318,  Cost : 0.916
    Step : 319,  Cost : 0.916
    Step : 320,  Cost : 0.916
    Step : 321,  Cost : 0.916
    Step : 322,  Cost : 0.916
    Step : 323,  Cost : 0.916
    Step : 324,  Cost : 0.916
    Step : 325,  Cost : 0.916
    Step : 326,  Cost : 0.916
    Step : 327,  Cost : 0.916
    Step : 328,  Cost : 0.916
    Step : 329,  Cost : 0.916
    Step : 330,  Cost : 0.916
    Step : 331,  Cost : 0.916
    Step : 332,  Cost : 0.916
    Step : 333,  Cost : 0.916
    Step : 334,  Cost : 0.916
    Step : 335,  Cost : 0.916
    Step : 336,  Cost : 0.916
    Step : 337,  Cost : 0.916
    Step : 338,  Cost : 0.916
    Step : 339,  Cost : 0.916
    Step : 340,  Cost : 0.916
    Step : 341,  Cost : 0.916
    Step : 342,  Cost : 0.916
    Step : 343,  Cost : 0.916
    Step : 344,  Cost : 0.916
    Step : 345,  Cost : 0.916
    Step : 346,  Cost : 0.916
    Step : 347,  Cost : 0.916
    Step : 348,  Cost : 0.916
    Step : 349,  Cost : 0.916
    Step : 350,  Cost : 0.916
    Step : 351,  Cost : 0.916
    Step : 352,  Cost : 0.916
    Step : 353,  Cost : 0.916
    Step : 354,  Cost : 0.916
    Step : 355,  Cost : 0.916
    Step : 356,  Cost : 0.916
    Step : 357,  Cost : 0.916
    Step : 358,  Cost : 0.916
    Step : 359,  Cost : 0.916
    Step : 360,  Cost : 0.916
    Step : 361,  Cost : 0.916
    Step : 362,  Cost : 0.916
    Step : 363,  Cost : 0.916
    Step : 364,  Cost : 0.916
    Step : 365,  Cost : 0.916
    Step : 366,  Cost : 0.916
    Step : 367,  Cost : 0.916
    Step : 368,  Cost : 0.916
    Step : 369,  Cost : 0.916
    Step : 370,  Cost : 0.916
    Step : 371,  Cost : 0.916
    Step : 372,  Cost : 0.916
    Step : 373,  Cost : 0.916
    Step : 374,  Cost : 0.916
    Step : 375,  Cost : 0.916
    Step : 376,  Cost : 0.916
    Step : 377,  Cost : 0.916
    Step : 378,  Cost : 0.916
    Step : 379,  Cost : 0.916
    Step : 380,  Cost : 0.916
    Step : 381,  Cost : 0.916
    Step : 382,  Cost : 0.916
    Step : 383,  Cost : 0.916
    Step : 384,  Cost : 0.916
    Step : 385,  Cost : 0.916
    Step : 386,  Cost : 0.916
    Step : 387,  Cost : 0.916
    Step : 388,  Cost : 0.916
    Step : 389,  Cost : 0.916
    Step : 390,  Cost : 0.916
    Step : 391,  Cost : 0.916
    Step : 392,  Cost : 0.916
    Step : 393,  Cost : 0.916
    Step : 394,  Cost : 0.916
    Step : 395,  Cost : 0.916
    Step : 396,  Cost : 0.916
    Step : 397,  Cost : 0.916
    Step : 398,  Cost : 0.916
    Step : 399,  Cost : 0.916
    Step : 400,  Cost : 0.916
    예측값 :  [0 1 0 0 0 0]
    실제값 :  [0 1 2 0 0 2]
    정확도 : 83.33
    

![image.png](attachment:image.png)

![image.png](attachment:image.png)


```python

```
