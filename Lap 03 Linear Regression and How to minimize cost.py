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