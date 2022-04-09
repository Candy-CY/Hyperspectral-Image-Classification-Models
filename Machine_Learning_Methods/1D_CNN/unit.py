import numpy as np
import tensorflow as tf
def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
def af(x):
    return tf.nn.relu(x)
def classifer_share1d(feature,class_num, cube_size=3,training=True, reuse=True):
    print(feature)
    feature = tf.expand_dims(feature, 4)
    f_num = 64
    with tf.variable_scope('classifer', reuse=reuse):
        with tf.variable_scope('conv00'):
            conv0 = tf.layers.conv3d(feature, f_num, (cube_size, 1, 8), strides=(1, 1, 3), padding='valid')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv01'):
            conv0 = tf.layers.conv3d(conv0, f_num, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv02'):
            conv0 = tf.layers.conv3d(conv0, f_num, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv10'):
            conv1 = tf.layers.conv3d(conv0, f_num * 2, (1, cube_size, 3), strides=(1, 1, 2), padding='valid')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv11'):
            conv1 = tf.layers.conv3d(conv1, f_num * 2, (1, 1, 3), strides=(1, 1, 2), padding='valid')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv12'):
            conv1 = tf.layers.conv3d(conv1, f_num * 2, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv20'):
            conv2 = tf.layers.conv3d(conv1, f_num * 4, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('conv21'):
            conv2 = tf.layers.conv3d(conv2, f_num * 4, (1, 1, 3), strides=(1, 1, 2), padding='valid')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('conv22'):
            conv2 = tf.layers.conv3d(conv2, f_num * 4, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('global_info'):
            f_shape = int(conv2.get_shape().as_list()[3])
            feature = tf.layers.conv3d(conv2, f_num * 8, (1, 1, f_shape), (1, 1, 1))
            feature = tf.layers.flatten(feature)
            print(feature)
        with tf.variable_scope('logits'):
            logtic = tf.layers.dense(feature,class_num)
    return logtic
