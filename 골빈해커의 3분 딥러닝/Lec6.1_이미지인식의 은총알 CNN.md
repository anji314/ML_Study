```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("./mnist/data/",one_hot=True)
```

    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    WARNING:tensorflow:From <ipython-input-1-bec8941499c3>:9: read_data_sets (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as: tensorflow_datasets.load('mnist')
    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:297: _maybe_download (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:299: _extract_images (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./mnist/data/train-images-idx3-ubyte.gz
    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:304: _extract_labels (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./mnist/data/train-labels-idx1-ubyte.gz
    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:112: _dense_to_one_hot (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.one_hot on tensors.
    Extracting ./mnist/data/t10k-images-idx3-ubyte.gz
    Extracting ./mnist/data/t10k-labels-idx1-ubyte.gz
    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:328: _DataSet.__init__ (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/_DataSet.py from tensorflow/models.
    


```python
X=tf.placeholder(tf.float32,[None,28,28,1])
Y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
```


```python
W1=tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
L1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
L1=tf.nn.relu(L1);
```


```python
L1=tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
```


```python
W2=tf.Variable(tf.random_normal([3,3,32, 64],stddev=0.01))
L2=tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
L2=tf.nn.relu(L2)
L2=tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
```


```python
W3=tf.Variable(tf.random_normal([7*7*64,256],stddev=0.01))
L3=tf.reshape(L2,[-1,7*7*64])
L3=tf.matmul(L3,W3)
L3=tf.nn.relu(L3)
L3=tf.nn.dropout(L3,keep_prob)
```

    WARNING:tensorflow:From <ipython-input-6-f5eef22deee6>:5: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    


```python
W4=tf.Variable(tf.random_normal([256,10],stddev=0.01))
model=tf.matmul(L3,W4)
```


```python
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=Y))
optimizer=tf.train.AdamOptimizer(0.01).minimize(cost)
```

    WARNING:tensorflow:From <ipython-input-8-bfec225fb70b>:1: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    


```python
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

batch_size=100
total_batch=int(mnist.train.num_examples/batch_size)

for epoch in range(15):
    total_cost=0
    
    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape(-1,28,28,1)
        
        _,cost_val=sess.run([optimizer,cost],
                           feed_dict={X:batch_xs,
                                     Y:batch_ys,
                                     keep_prob:0.7})
        total_cost+=cost_val
    
    print('Epoch : ','%04d'%(epoch+1),
         'Avg. cost = ','{:.3f}'.format(total_cost/total_batch))
print('최적화 완료!')


is_correct=tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))

print('정확도 : ',sess.run(accuracy,
                       feed_dict={X:mnist.test.images.reshape(-1,28,28,1),
                                 Y:mnist.test.labels,
                                 keep_prob:1}))
```

    Epoch :  0001 Avg. cost =  0.185
    Epoch :  0002 Avg. cost =  0.073
    Epoch :  0003 Avg. cost =  0.060
    Epoch :  0004 Avg. cost =  0.052
    Epoch :  0005 Avg. cost =  0.050
    Epoch :  0006 Avg. cost =  0.050
    Epoch :  0007 Avg. cost =  0.046
    Epoch :  0008 Avg. cost =  0.045
    Epoch :  0009 Avg. cost =  0.042
    Epoch :  0010 Avg. cost =  0.047
    Epoch :  0011 Avg. cost =  0.038
    Epoch :  0012 Avg. cost =  0.042
    Epoch :  0013 Avg. cost =  0.038
    Epoch :  0014 Avg. cost =  0.044
    Epoch :  0015 Avg. cost =  0.041
    최적화 완료!
    정확도 :  0.9877
    


```python

```
