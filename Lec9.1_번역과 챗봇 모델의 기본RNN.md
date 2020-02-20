```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("./mnist/data/",one_hot=True)


learning_rate=0.001
total_epoch=30
batch_size=128
n_input=28
n_step=28
n_hidden=128
n_class=10

```

    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    WARNING:tensorflow:From <ipython-input-1-d1829dbf0961>:5: read_data_sets (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
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
X=tf.placeholder(tf.float32,[None,n_step,n_input])
Y=tf.placeholder(tf.float32,[None,n_class])

W=tf.Variable(tf.random_normal([n_hidden,n_class]))
b=tf.Variable(tf.random_normal([n_class]))


```


```python
cell=tf.nn.rnn_cell.BasicRNNCell(n_hidden)
```

    WARNING:tensorflow:From <ipython-input-4-85222d2ccc4a>:1: BasicRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.SimpleRNNCell, and will be replaced by that in Tensorflow 2.0.
    


```python
outputs,states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
```


```python
outputs=tf.transpose(outputs,[1,0,2])
outputs=outputs[-1]
```


```python
model=tf.matmul(outputs,W)+b
```


```python
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=Y))

optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

    WARNING:tensorflow:From <ipython-input-10-e6e4f4b1d71c>:1: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    


```python
sess=tf.Session()
sess.run(tf.global_variables_initializer())

total_batch=int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost=0
    
    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape((batch_size,n_step,n_input))
        
        _,cost_val=sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys})
        total_cost+=cost_val
        
    print('Epoch : ','%04d'%(epoch+1),
         'Avg. cost = ','{:.3f}'.format(total_cost/total_batch))
    
print('최적화 완료!')





is_correct=tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))

test_batch_size=len(mnist.test.images)
test_xs=mnist.test.images.reshape(test_batch_size,n_step,n_input)
test_ys=mnist.test.labels

print('정확도 : ',sess.run(accuracy,feed_dict={X:test_xs,Y:test_ys}))
```

    Epoch :  0001 Avg. cost =  0.558
    Epoch :  0002 Avg. cost =  0.254
    Epoch :  0003 Avg. cost =  0.192
    Epoch :  0004 Avg. cost =  0.164
    Epoch :  0005 Avg. cost =  0.145
    Epoch :  0006 Avg. cost =  0.128
    Epoch :  0007 Avg. cost =  0.123
    Epoch :  0008 Avg. cost =  0.114
    Epoch :  0009 Avg. cost =  0.102
    Epoch :  0010 Avg. cost =  0.104
    Epoch :  0011 Avg. cost =  0.099
    Epoch :  0012 Avg. cost =  0.099
    Epoch :  0013 Avg. cost =  0.089
    Epoch :  0014 Avg. cost =  0.092
    Epoch :  0015 Avg. cost =  0.088
    Epoch :  0016 Avg. cost =  0.082
    Epoch :  0017 Avg. cost =  0.082
    Epoch :  0018 Avg. cost =  0.078
    Epoch :  0019 Avg. cost =  0.078
    Epoch :  0020 Avg. cost =  0.077
    Epoch :  0021 Avg. cost =  0.070
    Epoch :  0022 Avg. cost =  0.079
    Epoch :  0023 Avg. cost =  0.073
    Epoch :  0024 Avg. cost =  0.071
    Epoch :  0025 Avg. cost =  0.071
    Epoch :  0026 Avg. cost =  0.069
    Epoch :  0027 Avg. cost =  0.060
    Epoch :  0028 Avg. cost =  0.066
    Epoch :  0029 Avg. cost =  0.062
    Epoch :  0030 Avg. cost =  0.068
    최적화 완료!
    정확도 :  0.9665
    


```python

```
