import gzip
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

ejeX = []
ejeY = []

def mnist(hip):
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set
    x_data = train_x.astype('f4')  # the samples are the four first rows of data
    y_data = one_hot(train_y.astype(int), 10)  # the labels are in the last row. Then we encode them in one hot code
    x_valid_data = valid_x.astype('f4')
    y_valid_data = one_hot(valid_y.astype(int), 10)
    x_test_data = test_x.astype('f4')
    y_test_data = one_hot(test_y.astype(int), 10)
    x = tf.placeholder("float", [None, 784])  # samples de 0 a 785
    y_ = tf.placeholder("float", [None, 10])  # labels de 0 a 9
    W1 = tf.Variable(np.float32(np.random.rand(784, hip)) * 0.1)
    b1 = tf.Variable(np.float32(np.random.rand(hip)) * 0.1)
    W2 = tf.Variable(np.float32(np.random.rand(hip, 10)) * 0.1)
    b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)
    h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    # h = tf.matmul(x, W1) + b1  # Try this!
    y = tf.nn.softmax(tf.matmul(h, W2) + b2)
    loss = tf.reduce_sum(tf.square(y_ - y))
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    batch_size = 20
    epoch = 0
    error2 = 100
    error1 = 1000
    print("Training.......")
    while abs(error2 - error1) / float(error1) > 0.01:
        for jj in range(len(x_data) // batch_size):
            batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
            batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
            sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

        error1 = error2
        error2 = sess.run(loss, feed_dict={x: x_valid_data, y_: y_valid_data})
        # if epoch > 0:
        # ejeX.append(epoch)
        # ejeY.append(error1)
        # ejeY_V.append(error2)

        epoch = epoch + 1
    print("Validation...  ")
    result = sess.run(y, feed_dict={x: x_test_data})
    accuracy = 0
    for b, r in zip(y_test_data, result):
        if np.argmax(b) == np.argmax(r):
            accuracy += 1
    print("Accuracy with", hip, " was: ", accuracy / float(len(x_test_data)))
    per = accuracy / float(len(x_test_data))
    ejeY.append(per)
    print("----------------------------------------------------------------------------------")


mnist(1)
mnist(2)
mnist(5)
mnist(10)
mnist(20)
mnist(40)
mnist(60)
mnist(120)
ejeX.append(1.)
ejeX.append(2.)
ejeX.append(5.)
ejeX.append(10.)
ejeX.append(20.)
ejeX.append(40.)
ejeX.append(60.)
ejeX.append(120.)
plt.plot(ejeX, ejeY)
plt.title("Accuracy and Hip")
plt.xlabel("Hip")
plt.ylabel("Accuracy")
plt.show()
