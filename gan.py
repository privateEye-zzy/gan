import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(1)
np.random.seed(1)
BATCH_SIZE = 64
LENGTH = 5
Learning_G = 0.0001  # G网络的学习率
Learning_D = 0.0001  # D网络的学习率
ART_COMPONENTS = 15  # 画布的点数
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
def artist_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    return paintings
# G网络，根据噪点z生成新数据分布
with tf.variable_scope('Generator'):
    z = tf.placeholder(tf.float32, [None, LENGTH])
    G_l1 = tf.layers.dense(inputs=z, units=128, activation=tf.nn.relu)  # 全连接层
    G_out = tf.layers.dense(G_l1, ART_COMPONENTS)
# D网络，根据真实数据x和生成数据z来判别其分类
with tf.variable_scope('Discriminator'):
    # 真实数据的概率分布
    x = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in')
    D_l0 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu, name='l')
    prob_real = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')
    # G网络的概率分布
    D_l1 = tf.layers.dense(inputs=G_out, units=128,  activation=tf.nn.relu, name='l', reuse=True)
    prob_generate = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)
# 定义G和D网络的损失函数
D_loss = -tf.reduce_mean(tf.log(prob_real) + tf.log(1-prob_generate))
G_loss = tf.reduce_mean(tf.log(1-prob_generate))
# 定义G和D的优化器
train_D = tf.train.AdamOptimizer(Learning_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(Learning_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    epoch = 5000
    for step in range(epoch):
        input_x = artist_works()  # 生成真实数据
        noise_z = np.random.randn(BATCH_SIZE, LENGTH)  # 生成噪点数据
        G_paintings, prob_real_value, d_loss_value = sess.run([G_out, prob_real, D_loss, train_D, train_G],
                                           feed_dict={z: noise_z, x: input_x})[:3]
        if step % 50 == 0:
            print('iter:{}, d_loss:{}, d_accuracy:{}'.format(step, d_loss_value, prob_real_value.mean()))
            plt.cla()
            plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting', )
            plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_real_value.mean(), fontdict={'size': 15})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -d_loss_value, fontdict={'size': 15})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=12)
            plt.draw()
            plt.pause(0.01)
plt.ioff()
plt.show()
