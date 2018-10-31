import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist', one_hot=True)

batchSize = 128
totalBatch = mnist.train.num_examples // batchSize

trainEpochs = 20
# 参数概要？

# 初始化权值
def weightVariable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)
# 初始化偏置
def biasVariable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)
# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化层
def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 命名空间
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x_input")
    y = tf.placeholder(tf.float32, [None, 10], name="y_input")
    with tf.name_scope("x_image"):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

with tf.name_scope("Conv1"):
    with tf.name_scope("W_conv1"):
        W_conv1 = weightVariable([5, 5, 1, 32], name="W_conv1")
    with tf.name_scope("b_conv1"):
        b_conv1 = biasVariable([32], name="b_conv1")
    with tf.name_scope("conv2d_1"):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope("relu"):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope("h_pool1"):
        h_pool1 = maxPool(h_conv1)

with tf.name_scope("Conv2"):
    with tf.name_scope("W_conv2"):
        W_conv2 = weightVariable([5, 5, 32, 64], name="W_conv2")
    with tf.name_scope("b_conv2"):
        b_conv2 = biasVariable([64], name="b_conv2")

    with tf.name_scope("conv2d_2"):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope("relu"):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope("h_pool2"):
        h_pool2 = maxPool(h_conv2)


with tf.name_scope("fc1"):
    with tf.name_scope("W_fc1"):
        W_fc1 = weightVariable([7*7*64, 1024], name="W_fc1")
    with tf.name_scope("b_fc1"):
        b_fc1 = biasVariable([1024], name="b_fc1")

    with tf.name_scope("h_pool2_flat"):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name="h_pool2_flat")
    with tf.name_scope("wx_plus_b1"):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope("relu"):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    with tf.name_scope("keep_prob"):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    with tf.name_scope("h_fc1_drop"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

with tf.name_scope("fc_2"):
    with tf.name_scope("W_fc2"):
        W_fc2 = weightVariable([1024, 10], name="W_fc2")
    with tf.name_scope("b_fc2"):
        b_fc2 = biasVariable([10], name="b_fc2")

    with tf.name_scope("xw_plus_b2"):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope("prediction"):
        prediction = wx_plus_b2

with tf.name_scope("lossFunc"):
    lossAll = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    loss = tf.reduce_mean(lossAll)
    tf.summary.scalar("lossFunc", loss)

# Optimizer
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.name_scope("accuracy"):
    with tf.name_scope("correctPrediction"):
        correctPrediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    trainWriter = tf.summary.FileWriter("./logs/train", sess.graph)
    testWriter = tf.summary.FileWriter("./logs/test", sess.graph)

    for epoch in range(trainEpochs):
        print("=== Epoch {:0>4} ===".format(epoch+1))
        for i in range(totalBatch):
            print("\n\t=== Iter {:0>4} ===".format(i+1))
            batchTrainX, batchTrainY = mnist.train.next_batch(batchSize)
            batchTestX, batchTestY = mnist.test.next_batch(batchSize)
            _, ls = sess.run([train_step, loss], feed_dict={x: batchTrainX, y: batchTrainY, keep_prob: 0.5})
            trainAccuracy = sess.run(accuracy, feed_dict={x: batchTrainX, y: batchTrainY, keep_prob: 0.5})
            testAccuracy = sess.run(accuracy, feed_dict={x: batchTestX, y: batchTestY, keep_prob: 0.5})
            # summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            # train_writer.add_summary(summary, i)
            #
            # batch_xs, batch_ys = mnist.test.next_batch(batchSize)
            # summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            # test_writer.add_summary(summary, i)

            print("\n\t\tLoss={:.4f} \tTraining accuracy={:.4f} \tTesting accuracy={:.4f}".format(ls, trainAccuracy, testAccuracy))

        summary = sess.run(merged, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 0.5})
        trainWriter.add_summary(summary, epoch)
        summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 0.5})
        testWriter.add_summary(summary, epoch)

        trainEpochAccuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 0.5})
        testEpochAccuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 0.5})
        epochLoss = sess.run(loss, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 0.5})

        print("Average loss, training accuracy and testing accuracy of every epoch:")
        print("\n\tLoss={:.4f} \tTraining accuracy={:.4f} \tTesting accuracy={:.4f}\n\n".format(epochLoss, trainEpochAccuracy, testEpochAccuracy))
