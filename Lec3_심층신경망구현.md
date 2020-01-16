```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
```

    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    


```python
x_data=np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])

y_data=np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,0,1],
])

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
```


```python
W1=tf.Variable(tf.random_uniform([2,10],-1.,1.))
W2=tf.Variable(tf.random_uniform([10,3],-1.,1.))

b1=tf.Variable(tf.zeros([10]))
b2=tf.Variable(tf.zeros([3]))
```


```python
L1=tf.add(tf.matmul(X,W1),b1)
L1=tf.nn.relu(L1)
```


```python
model=tf.add(tf.matmul(L1,W2),b2)
```


```python
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=model))

optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
train_op=optimizer.minimize(cost)
```

    WARNING:tensorflow:From <ipython-input-6-7776e773714e>:1: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    


```python
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
```


```python
for step in range(100):
    sess.run(train_op,feed_dict={X:x_data,Y:y_data})
    
    if(step+1)%10==0:
        print(step+1,sess.run(cost,feed_dict={X:x_data,Y:y_data}))
```

    10 0.041670714
    20 0.03313409
    30 0.02700386
    40 0.0224806
    50 0.019036537
    60 0.016350899
    70 0.014215402
    80 0.01248631
    90 0.01106369
    100 0.009879804
    


```python
prediction=tf.argmax(model,1)
target=tf.argmax(Y,1)

print('예측값 : ',sess.run(prediction,feed_dict={X:x_data}))
print('실제값 : ',sess.run(target,feed_dict={Y:y_data}))
```

    예측값 :  [0 1 2 0 0 2]
    실제값 :  [0 1 2 0 0 2]
    


```python
is_correct=tf.equal(prediction,target)
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도 : %.2f' %sess.run(accuracy*100,feed_dict={X:x_data,Y:y_data}))
```

    정확도 : 100.00
    


```python

```
