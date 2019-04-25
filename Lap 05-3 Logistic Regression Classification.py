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