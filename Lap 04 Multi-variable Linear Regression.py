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