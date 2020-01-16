```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

```

    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    


```python
import numpy as np

```


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
```


```python
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W=tf.Variable(tf.random_uniform([2,3],-1.,1.))
b=tf.Variable(tf.zeros([3]))

```


```python
L=tf.add(tf.matmul(X,W),b)
L=tf.nn.relu(L)
```


```python
model=tf.nn.softmax(L)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model),axis=1))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op=optimizer.minimize(cost)
```


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

    10 1.0342299
    20 1.0334364
    30 1.0326084
    40 1.0318333
    50 1.0310241
    60 1.0302669
    70 1.0294901
    80 1.0287356
    90 1.0279844
    100 1.0272382
    


```python
prediction=tf.argmax(model,axis=1)
target=tf.argmax(Y,axis=1)
print('예측값 : ',sess.run(prediction,feed_dict={X:x_data}))
print('실제값 : ',sess.run(target,feed_dict={Y:y_data}))
```

    예측값 :  [0 1 1 0 0 0]
    실제값 :  [0 1 2 0 0 2]
    


```python
is_correct=tf.equal(prediction,target)
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도: %.2f' %sess.run(accuracy*100,feed_dict={X:x_data,Y:y_data}))
```

    정확도: 66.67
    


```python

```


```python

```
