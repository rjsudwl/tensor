#모두를 위한 딥러닝 시즌2 Lab-02 Simple Liner Regression LAB
import tensorflow as tf   #텐서플로우 실행
tf.enable_eager_execution() # execution 활성화하여 즉시 실행

# Data
x_data = [1, 2, 3, 4, 5] # x 데이터를 넘겨줌
y_data = [1, 2, 3, 4, 5] # y 데이터를 넘겨줌

# W, b initialize
W = tf.Variable(2.9) # W에 임의의 값 지정
b = tf.Variable(0.5) # b에 임의의 값 지정

learning_rate = 0.01 #학습 속도로 0.01의 상수값을 줌

for i in range(100+1): # W, b update # 학습을 100번 반복, W와 b의 값을 지속적으로 갱신
    # Gradient descent
    with tf.GradientTape() as tape: #경사하강법
        hypothesis = W * x_data + b
    cost = tf.reduce_mean(tf.square(hypothesis - y_data)) #가설과 실제 값의 차이의 평균으로 cost값 지정
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad) #assign_sub 메소드를 이용하여 W값에 대한 할당을 변경
    b.assign_sub(learning_rate * b_grad) #assign_sub 메소드를 이용하여 b값에 대한 할당을 변경
    if i % 10 == 0: #W와 b 값의 변화를 확인하기 위해 10번에 한번씩 출력
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

# Lab-03 Liner Regression and How to minimize cost LAB

# Cost function in pure Python
import numpy as np # 라이브러리를 np라는 이름으로 import

X = np.array([1, 2, 3]) #랭크가 1인 배열의 데이터 생성
Y = np.array([1, 2, 3])

def cost_func(W, X, Y): #cost함수 값을 구하는 수식을 표현
    c = 0
    for i in range(len(X)):  # X의 길이만큼 반복
        c += (W * X[i] - Y[i]) ** 2 #편차 제곱의 합계를 c에 저장
    return c / len(X)    #갯수 m에 해당하는 길이 X로 c를 나누면서 평균구하기

for feed_W in np.linspace(-3, 5, num=15): # linspace함수를 이용하여 -3에서 5까지 15개의 구간
    curr_cost = cost_func(feed_W, X, Y)  # feed_W값에 따라 바뀌는 cost 값의 변화를 출력
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost)) #

# Gradient descent
tf.set_random_seed(0) # for reproducibility # random_seed를 초기화

x_data = [1., 2., 3., 4.] # x와 y 데이터값 지정
y_data = [1., 3., 5., 7.]

W = tf.Variable(tf.random_normal([1], -100., 100.)) #정규분포를 따르는 random 수를 한개짜리로 변수를 만들어 W로 지정

for step in range(300):     # for문을 이용하여 gradient descent부분을 300회 수행
    hypothesis = W * X      #가설 설정
    cost = tf.reduce_mean(tf.square(hypothesis - Y))  #실제 값과의 차이 제곱의 평균을 내는 것으로 cost함수를 설정

    #gradient descent의 수식을 설명
    alpha = 0.01   #상수 값 지정
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))  # W와 X를  곱한것에 Y를 빼고 X를 곱한 것으로 평균
    descent = W - tf.multiply(alpha, gradient) #상수 값과 gradient식을 곱하여 경사하강식
    W.assign(descent) #하강

    if step % 10 == 0: #10번에 한번씩 확인
        print('{:5} | {:10.4f} | {:10.6f}'.format(
            step, cost.numpy(), W.numpy()[0])) # cost값과 W값 변화 출력

# Lab-04 Multi-variable linear regression LAB

# random_seed로 초기화
tf.set_random_seed(0)  # for reproducibility

# data and label # X데이터와 Y값 지정
x1 = [ 73., 93., 89., 96., 73.]
x2 = [ 80., 88., 91., 98., 66.]
x3 = [ 75., 93., 90., 100., 70.]
Y = [152., 185., 180., 196., 142.]

# random weights # X만큼 변수 3개를 만들고 초기값을 1로 설정
w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
w3 = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001 #학습속도로 0.000001을 지정

for i in range(1000+1): # 1001회 수행
    # tf.GradientTape() to record the gradient of the cost function
    with tf.GradientTape() as tape: # 아래의 변수들을 gradientTape에 저장
        hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b #W와 X의 곱과 b를 더하는 가설
        cost = tf.reduce_mean(tf.square(hypothesis - Y)) # cost값 정하기
    # calculates the gradients of the cost
    #위에 저장해둔 tape에서의 값들을 w1_grad, w2_grad, w3_grad, b_grad에 각각 할당
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

    # update w1,w2,w3 and b
    #각각의 값을 assign_sub을 통해 learning rate와 w_grad를 곱하고 이를 빼서 지속적으로 update
    w1.assign_sub(learning_rate * w1_grad)
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0: #50번마다 cost값 출력
    print("{:5} | {:12.4f}".format(i, cost.numpy()))
# Matrix
#데이터를 column별로 간결하게 표현
data = np.array([
    # X1, X2, X3, y
    [ 73., 80., 75., 152. ],
    [ 93., 88., 93., 185. ],
    [ 89., 91., 90., 180. ],
    [ 96., 98., 100., 196. ],
    [ 73., 66., 70., 142. ]
], dtype=np.float32)

# slice data
# 데이터를 slice하여 X와 Y의 범위를 주어 데이터값 설정
X = data[:, :-1] # 마지막 column을 제외한 5행 3열의 데이터
y = data[:, [-1]] #마지막 column

# X의 column의 경우 3개이므로 W의 값도 3개를 준비하여 출력값 1개로 지정
W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001 # 학습속도 지정

# hypothesis, prediction function
def predict(X): # 예측함수 다음과 같이 지정
    return tf.matmul(X, W) + b

n_epochs = 2000
for i in range(n_epochs+1): #2001번의 순회
    # record the gradient of the cost function
    with tf.GradientTape() as tape: # cost를 Tape에 저장
        cost = tf.reduce_mean((tf.square(predict(X) - y)))

# calculates the gradients of the loss
W_grad, b_grad = tape.gradient(cost, [W, b]) #tape을 불러내어 변수할당

# updates parameters (W and b)
# 변수를 학습속도와 곱하여 지속적으로 할당변화 update
W.assign_sub(learning_rate * W_grad)
b.assign_sub(learning_rate * b_grad)

if i % 100 == 0: #100번에 한번 cost값 출력
    print("{:5} | {:10.4f}".format(i, cost.numpy()))

# Lab-05-2 logistic_regression

import tensorflow.contrib.eager as tfe # tensorflow를 eager모드로 실행하기위한 라이브러리 import
tf.enable_eager_execution() # eager모드 실행을 위한 execution 선언
# tf.data를 통해 x,y데이터를 실제 x의 길이만큼 배치하기위한 data값 불러오기
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
W = tf.Variable(tf.zeros([2,1]), name='weight')  # 2행 1열의 값
b = tf.Variable(tf.zeros([1]), name='bias')
def logistic_regression(features):
    hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b)) #sigmoid함수를 사용하여 가설설정
    return hypothesis
def loss_fn(hypothesis, labels): #가설과 label값을 통해 cost값 구현
    cost = -tf.reduce_mean(labels * tf.log(loss_fn(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))
    return cost
def grad(hypothesis, features, labels): #학습을 위해 가설과 label을 통해 나온 loss값을 선언
    with tf.GradientTape() as tape:
        loss_value = loss_fn(hypothesis,labels)
    return tape.gradient(loss_value, [W,b]) #gradient를 통해 지속적으로 값을 변환
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

for step in range(EPOCHS): #데이터값으로 feature와 label값을 가설을 통해 grads값을 설정하여 minimize
    for features, labels in tfe.Iterator(dataset):
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b])) #cost값을 최소화
        if step % 100 == 0: #100번에 한번 출력 step과 loss, label값
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features) ,labels)))

def accuracy_fn(hypothesis, labels): #가설과 실제값 비교
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) #예측한 값
    # 예측값과 실제값을 비교하여 평균을 낸 값을 accuracy로 출력
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

#x값과 y값을 통해 정확도 test
test_acc = accuracy_fn(logistic_regression(x_test),y_test)