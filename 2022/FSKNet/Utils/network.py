import tensorflow as tf
import numpy as np
import time
from data import Data

class Network(object):
    """
    CVPR2017: Neural Aggregation Network for Video Face Recognition Aggregation module
    """

    # ( batch_size, 512, class_num, 5 )
    def __init__(self, batch_size, feature_len, class_num, group):
        """
        batch_size: batch size 
        feature_len: input feature length
        class_num: class number
        """
        self.batch_size = batch_size
        self.feature_len = feature_len
        self.class_num = class_num
        self.group = group

    def create_network(self, input_x):

        w1 = tf.get_variable("fc1/weights", shape=[self.feature_len, self.feature_len],
                             initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-4))
        b1 = tf.get_variable("fc1/biases", shape=[self.feature_len], initializer=tf.constant_initializer(0.0001))
        w2 = tf.get_variable("fc2/weights", shape=[self.feature_len, self.class_num],
                             initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-4))
        b2 = tf.get_variable("fc2/biases", shape=[self.class_num], initializer=tf.constant_initializer(0.0001))
        q_param = tf.get_variable("q0", shape=[self.feature_len], initializer=tf.constant_initializer(0.0001))

        # attention module 1
        resize_input = tf.reshape(input_x, [self.batch_size * self.group, self.feature_len])
        expand_param = tf.expand_dims(q_param, 1)
        temp = tf.matmul(resize_input, expand_param)
        temp = tf.reshape(temp, [self.batch_size, self.group])
        temp = tf.nn.softmax(temp)
        features = tf.split(axis=0, num_or_size_splits=self.batch_size, value=input_x)
        temps = tf.split(axis=0, num_or_size_splits=self.batch_size, value=temp)
        fusion = [tf.matmul(temps[i], features[i][0]) for i in range(self.batch_size)]
        r1 = tf.concat(axis=0, values=fusion)

        # fc1 layer
        fc = tf.add(tf.matmul(r1, w1), b1, name="fc1")
        tanh = tf.nn.tanh(fc)

        # attention module 2
        input_split = tf.split(axis=0, num_or_size_splits=self.batch_size, value=input_x)
        q1_split = tf.split(axis=0, num_or_size_splits=self.batch_size, value=tanh)
        a1 = [tf.tensordot(features[i], q1_split[i], 1) for i in range(self.batch_size)]
        a1_fusion = tf.concat(axis=0, values=a1)
        e1 = tf.nn.softmax(a1_fusion)
        temp1 = tf.split(axis=0, num_or_size_splits=self.batch_size, value=e1)
        fusion1 = [tf.matmul(temps[i], features[i][0]) for i in range(self.batch_size)]
        r2 = tf.concat(axis=0, values=fusion)

        # fc2 layer
        predict = tf.add(tf.matmul(r2, w2), b2, name="predict")
        return r2, predict

    def train_network(self, epoch, filename):
        """
        """
        input_x = tf.placeholder(tf.float32, shape=[self.batch_size, self.group, self.feature_len])
        label_x = tf.placeholder(tf.int32, shape=[self.batch_size, self.class_num])
        _, predict = self.create_network(input_x)

        dataset = Data(filename, self.batch_size, self.class_num)
        dataset.load_feature()

        static = tf.equal(tf.argmax(predict, 1), tf.argmax(label_x, 1))
        accuracy = tf.reduce_mean(tf.cast(static, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_x, logits=predict)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", loss)

        optim = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/", sess.graph)

        for i in range(epoch):
            feature_x, labels_x = dataset.next_batch(self.group)
            _ = sess.run([optim], feed_dict={input_x: feature_x, label_x: labels_x})
            if i % 10 == 0:
                _acc, _loss, results = sess.run([accuracy, loss, merged],
                                                feed_dict={input_x: feature_x, label_x: labels_x})
                print("%s\tIteration\t%d\tAccuracy\t%f\tLoss\t%f" % (time.asctime(), i, _acc.item(), _loss.item()))
                writer.add_summary(results, i)

            if i % (epoch / 5) == 0:
                saver.save(sess, "./model/attention.ckpt", global_step=i)

if __name__ == "__main__":
    filename = "./YoutubeFaces.mat"
    batch_size = 128
    class_num = 1595
    net = Network(batch_size, 512, class_num, 5)
    net.train_network(1000000, filename)
