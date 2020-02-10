## 4.1 인공 신경망의 작동 원리

인공 신경망은 뉴런(뇌를 구성하는 신경세포)의 동작 원리에 기초한다.
받은 자극이 전달하기에 충분히 강하면 다음 뉴런 전달하고, 아니면 전달하지 않는다.

#### 즉, 입력신호(== 입력값 X에 가중치(W)를 곱하고 편향(b)를 더한뒤 활성화 함수를 거쳐서 결괏값 y를 만들어 낸다.
- 활성화 함수 : 인공 뉴런의 핵심 요소. 대표적으로 Sigmoid, ReLU, tanh 함수가 있다.


인공 뉴런은 가중치와 활성화 함수로 연결된 매우 간단한 구조이지만, 인공 뉴런을 충분히 많이 연결해 놓으면 매우 복잡한 패턴까지도 스스로 학습 할수 있게 된다.

- 신경망의 기본 학습 방법 : 모든 조합의 경우의 수에 대해 가중치를 대입하고 계산한다.(입력층 부터 가중치를 조절해 가는 방식)
- 역전파 :  결과값의 오차를 앞쪽으로 전파하면서 가중치를 갱신한다.

- - -
## 4.2 간단한 분류 모델 구현하기

이진 데이터를 이용하여 여러 종류로 구분하는 분류를 해보자.




```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

```

    WARNING:tensorflow:From c:\users\안지혜\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    
### 1) NumPy 라이브러리 임포트
- NumPy 라이브러리 : 수치해석용 차이썬 라이브러리. 행렬 조작과 연산에 필수 적이다.
```python
import numpy as np

```

[털,날개]  -> 있으면 1, 없으면 0
```python
x_data=np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])
```
[기타,포유류,조류] 
- 원-핫 인코딩 :표현하고자하는 값을 뜻하는 인덱스의 원소만 1로 표기하고 나머지 원소는 모두 0으로 채우는 표기법. 
```
y_data=np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,0,1],
])
```




### 2) 신경망 모델을 구성
```python
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
```
- 가중치 변수 W는 [입력층(특징 수),출력 층(레이블 수)]의 구성 : [2,3]
- 편향 변수 b는 레이블 수인 3개의 요소를 가진 변수로 설정. : [3]

이 가중치를 곱하고 편향을 더한 결과를 활성화 함수인 ReLU에 적용하면 신경망 구성 완료.
```python
W=tf.Variable(tf.random_uniform([2,3],-1.,1.))
b=tf.Variable(tf.zeros([3]))


L=tf.add(tf.matmul(X,W),b)
L=tf.nn.relu(L)
```

### 3) 손실 함수 작성
- softmax 함수 : 배열 내의 결괏 값들을 전체 합이 1이 되도록 만든다. -> 해당 결과를 확률로 해석이 가능하다.ex) [0.2,0.7,0.1]
- 교차 엔트로피 함수 : 예측값과 실제값 사이의 확률 분포 차이를 계산한다. (손실 함수)
```python
model=tf.nn.softmax(L)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model),axis=1))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op=optimizer.minimize(cost)
```
- reduce_XXX 함수: 텐서의 차원을 줄여준다. XXX부분이 구체적인 차원 축소 방법을 뜻하고, axis 매개변수는 축소할 차원을 정한다. 




### 4) 학습
```python
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
```
학습을 백전 진행, 학습 도중 10번에 한번씩 손실 갑을 출력.
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
    

### 5) 학습된 결과
-argmax 함수 : 요소 중 가장 큰 값의 인덱스를 찾아준다.(model을 바로 출력하면 (ex : [0.2, 0.7, 0.1])과 값이 나온다.)
```python
prediction=tf.argmax(model,axis=1)
target=tf.argmax(Y,axis=1)
print('예측값 : ',sess.run(prediction,feed_dict={X:x_data}))
print('실제값 : ',sess.run(target,feed_dict={Y:y_data}))
```

    예측값 :  [0 1 1 0 0 0]
    실제값 :  [0 1 2 0 0 2]
    

### 6) 정확도를 출력
```python
is_correct=tf.equal(prediction,target)
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도: %.2f' %sess.run(accuracy*100,feed_dict={X:x_data,Y:y_data}))
```

    정확도: 66.67
    


여러번 재실행을 시켜도 정확도가 80이상 처럼 높아지지 않는다ㅜㅜ
이유 : 신경망이 한 층밖에 안되기 때문이다.
