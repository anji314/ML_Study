```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
```

    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    


```python
char_arr=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

num_dic={n:i for i,n in enumerate(char_arr)}
dic_len=len(num_dic)

```


```python
seq_data=['word','wood','deep','dive','cold','cool','load','love','kiss','kind']

```


```python
def make_batch(seq_data):
    input_batch=[]
    target_batch=[]
    
    
    for seq in seq_data:
        input=[num_dic[n] for n in seq[:-1]]
        target=num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)
        
    return input_batch, target_batch

```


```python
learning_rate=0.01
n_hidden=128
total_epoch=30

n_step=3
n_input=n_class=dic_len

```


```python
X=tf.placeholder(tf.float32,[None,n_step,n_input])
Y=tf.placeholder(tf.int32,[None])

W=tf.Variable(tf.random_normal([n_hidden,n_class]))
b=tf.Variable(tf.random_normal([n_class]))
```


```python
cell1=tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1=tf.nn.rnn_cell.DropoutWrapper(cell1,output_keep_prob=0.5)
cell2=tf.nn.rnn_cell.BasicLSTMCell(n_hidden)


```

    WARNING:tensorflow:From <ipython-input-7-dcb220d7db58>:1: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
    


```python
multi_cell=tf.nn.rnn_cell.MultiRNNCell([cell1,cell2])

outputs,states=tf.nn.dynamic_rnn(multi_cell,X,dtype=tf.float32)

```

    WARNING:tensorflow:From <ipython-input-8-7415ad5f096d>:1: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From <ipython-input-8-7415ad5f096d>:3: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\ops\rnn_cell_impl.py:735: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.add_weight` method instead.
    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\ops\rnn_cell_impl.py:739: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    


```python
outputs=tf.transpose(outputs,[1,0,2])
outputs=outputs[-1]
model=tf.matmul(outputs,W)+b
```


```python
cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model,labels=Y))
```


```python
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
```


```python
sess=tf.Session()
sess.run(tf.global_variables_initializer())

input_batch,target_batch=make_batch(seq_data)

for epoch in range(total_epoch):
    _,loss=sess.run([optimizer,cost],feed_dict={X:input_batch,Y:target_batch})
    
    print('Epoch : ','%04d'%(epoch+1),'cost = ','{:.6f}'.format(loss))
    
print('최적화 완료!')
```

    Epoch :  0001 cost =  3.866539
    Epoch :  0002 cost =  2.914603
    Epoch :  0003 cost =  1.828688
    Epoch :  0004 cost =  1.451979
    Epoch :  0005 cost =  0.854312
    Epoch :  0006 cost =  0.892454
    Epoch :  0007 cost =  0.354306
    Epoch :  0008 cost =  0.643901
    Epoch :  0009 cost =  0.507177
    Epoch :  0010 cost =  0.387260
    Epoch :  0011 cost =  0.161664
    Epoch :  0012 cost =  0.316158
    Epoch :  0013 cost =  0.325205
    Epoch :  0014 cost =  0.233549
    Epoch :  0015 cost =  0.188036
    Epoch :  0016 cost =  0.090804
    Epoch :  0017 cost =  0.127499
    Epoch :  0018 cost =  0.110506
    Epoch :  0019 cost =  0.175751
    Epoch :  0020 cost =  0.125495
    Epoch :  0021 cost =  0.184909
    Epoch :  0022 cost =  0.041198
    Epoch :  0023 cost =  0.062918
    Epoch :  0024 cost =  0.109037
    Epoch :  0025 cost =  0.069734
    Epoch :  0026 cost =  0.040914
    Epoch :  0027 cost =  0.049227
    Epoch :  0028 cost =  0.078092
    Epoch :  0029 cost =  0.024325
    Epoch :  0030 cost =  0.102441
    최적화 완료!
    


```python
prediction=tf.cast(tf.argmax(model,1),tf.int32)
prediction_check=tf.equal(prediction,Y)
accuracy=tf.reduce_mean(tf.cast(prediction_check,tf.float32))

input_batch,target_batch=make_batch(seq_data)

predict,accuracy_val=sess.run([prediction,accuracy],feed_dict={X:input_batch,Y:target_batch})

predict_words=[]
for idx,val in enumerate(seq_data):
    last_char=char_arr[predict[idx]]
    predict_words.append(val[:3]+last_char)
    
print('\n===예측결과===')
print('입력값 : ',[w[:3]+' ' for w in seq_data])
print('예측값 : ',predict_words)
print('정확도 : ',accuracy_val)
```

    
    ===예측결과===
    입력값 :  ['wor ', 'woo ', 'dee ', 'div ', 'col ', 'coo ', 'loa ', 'lov ', 'kis ', 'kin ']
    예측값 :  ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
    정확도 :  1.0
    


```python

```
