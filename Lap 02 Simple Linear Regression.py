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