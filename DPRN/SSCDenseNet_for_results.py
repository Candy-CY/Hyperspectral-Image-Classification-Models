import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time

samples_type=['ratio','same_num'][0]

for (FLAG,curr_train_ratio) in [(1,0.1)]:
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Seed_List=[0,1,2,3,4]#随机种子点
    # SeFLAGed_List=[0]#随机种子点
    if FLAG == 1:
        data_mat = sio.loadmat('..\\HyperImage_data\\indian\\Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('..\\HyperImage_data\\indian\\Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
        # 参数预设
        train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 1000  # 迭代次数
        dataset_name = "indian_SSCDenseNet"  # 数据集名称
        pass
    if FLAG == 2:
        data_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\Pavia_University_gt.mat')
        gt = gt_mat['pavia_university_gt']
        # 参数预设
        train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 9  # 样本类别数
        learning_rate = 1e-3  # 学习率
        max_epoch = 1000  # 迭代次数
        dataset_name = "paviaU_SSCDenseNet"  # 数据集名称
        pass
    if FLAG == 3:
        data_mat = sio.loadmat('..\\HyperImage_data\\Salinas\\Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('..\\HyperImage_data\\Salinas\\Salinas_gt.mat')
        gt = gt_mat['salinas_gt']
        # 参数预设
        train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 1e-3  # 学习率
        max_epoch = 1000  # 迭代次数
        dataset_name = "salinas"  # 数据集名称
        pass
    if FLAG == 4:
        data_mat = sio.loadmat('..\\HyperImage_data\\Simu\\Simu_data.mat')
        data = data_mat['Simu_data']
        gt_mat = sio.loadmat('..\\HyperImage_data\\Simu\\Simu_label.mat')
        gt = gt_mat['Simu_label']
        # 参数预设
        train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 5  # 样本类别数
        learning_rate = 1e-3  # 学习率
        max_epoch = 1000  # 迭代次数
        dataset_name = "simu"  # 数据集名称
        pass
    if FLAG == 5:
        data_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC.mat')
        data = data_mat['KSC']
        gt_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC_gt.mat')
        gt = gt_mat['KSC_gt']
        # 参数预设
        train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 13  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 1000  # 迭代次数
        dataset_name = "KSC_SSCDenseNet"  # 数据集名称
        pass
    ###########
    train_samples_per_class=curr_train_ratio
    #当定义为每类样本个数时,则该参数更改为训练样本数
    val_samples=class_count
    train_ratio=curr_train_ratio
    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data.shape  # 高光谱数据的三个维度
    #打印每类样本个数
    # gt_reshape=np.reshape(gt, [-1])
    # for i in range(class_count):
    #     idx = np.where(gt_reshape == i + 1)[-1]
    #     samplesCount = len(idx)
    #     print(samplesCount)
    
    def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
        '''
        get classification map , then save to given path
        :param label: classification label, 2D
        :param name: saving path and file's name
        :param scale: scale of image. If equals to 1, then saving-size is just the label-size
        :param dpi: default is OK
        :return: null
        '''
        fig, ax = plt.subplots()
        numlabel = np.array(label)
        v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        foo_fig = plt.gcf()  # 'get current figure'
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
        pass
    for curr_seed in Seed_List:
        # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
        random.seed(curr_seed)
        gt_reshape = np.reshape(gt, [-1])
        train_rand_idx = []
        val_rand_idx = []
        if samples_type=='ratio':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)
            train_rand_idx = np.array(train_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            ##将测试集（所有样本，包括训练样本）也转化为特定形式
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)
            # 背景像元的标签
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx
            # 从测试集中随机选取部分样本作为验证集
            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
            # 将训练集 验证集 测试集 整理
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)
        if samples_type=='same_num':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                real_train_samples_per_class=train_samples_per_class
                rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                if real_train_samples_per_class>samplesCount:
                    real_train_samples_per_class=samplesCount
                    # val_samples_per_class=0
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)
                # if val_samples_per_class>0:
                #     rand_real_idx_per_class_val = idx[rand_idx[-val_samples_per_class:]]
                #     val_rand_idx.append(rand_real_idx_per_class_val)
            train_rand_idx = np.array(train_rand_idx)
            val_rand_idx = np.array(val_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)

            train_data_index = set(train_data_index)
            # val_data_index = set(val_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)

            # 背景像元的标签
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index  - background_idx
            # 从测试集中随机选取部分样本作为验证集
            val_data_count = int(val_samples)  # 验证集数量
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index=test_data_index-val_data_index
            # 将训练集 验证集 测试集 整理
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)

        # 获取训练样本的标签图
        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass
        
        # 获取测试样本的标签图
        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass
        
        Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图
        
        # 获取验证集样本的标签图
        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass

        # 获取对应形状的权重w和偏差b，带默认初始化操作
        def get_weight_variable(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

        def get_bias_variable(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

        # 卷积操作和池化操作
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def conv2d_depthwise(x, W):
            return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def model(last_layers_chanels: int, spatial_block_chanels: int, data, gt):
            global m, n, d, class_count
            last_layers_chanels = 150
            spatial_block_chanels = 50
            
            # 定义BN正则化函数
            def BN(layer):
                h_temp = tf.layers.batch_normalization(layer, training=True)
                return h_temp
            
            def SSConv(input, output_dims):
                a = BN(input)
                b = tf.layers.conv2d(a, output_dims, 1, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0003))  #
                b = tf.nn.sigmoid(b)
                w_conv1 = get_weight_variable([5, 5, output_dims, 1])
                b_conv1 = get_bias_variable([output_dims])
                c = conv2d_depthwise(b, w_conv1) + b_conv1
                c = tf.nn.sigmoid(c)  # 128
                return c
            #tf.placeholder()函数作为一种占位符用于定义过程，可以理解为形参，在执行的时候再赋具体的值。
            input_img = tf.placeholder(tf.float32, [1, m, n, d])
            real_label = tf.placeholder(tf.float32, shape=[1, m, n, class_count])
            real_label_mask = tf.placeholder(tf.float32, shape=[1, m, n, class_count])  # 用于掩膜
            drop = tf.placeholder(tf.float32)  # dropout
            
            # 第一层
            h_conv1 = SSConv(input_img, last_layers_chanels)
            
            # 第二层
            h_conv2 = SSConv(h_conv1, spatial_block_chanels)
            h_conv2 = tf.concat([h_conv2, h_conv1], axis=-1)  # 拼接后的维度应该为203维
            
            # 第三层 spatial block
            h_conv3 = SSConv(h_conv2, spatial_block_chanels)
            h_conv3 = tf.concat([h_conv3, h_conv2], axis=-1)  # 拼接后的维度应该为448维
            
            # 第四层 spatial block
            h_conv4 = SSConv(h_conv3, spatial_block_chanels)
            h_conv4 = tf.concat([h_conv4, h_conv3], axis=-1)  # 拼接后的维度应该为480维
            
            # 第五层 spatial block
            h_conv5 = SSConv(h_conv4, spatial_block_chanels)
            h_conv5 = tf.concat([h_conv5, h_conv4], axis=-1)  # 拼接后的维度应该为496维
            
            # 第六层 spatial block
            h_conv6 = SSConv(h_conv5, spatial_block_chanels)
            
            h_conv6 = tf.concat([h_conv6, h_conv5], axis=-1)  # 拼接后的维度应该为504维
            # h_conv6 = tf.layers.max_pooling2d(h_conv6, [5, 5], [1, 1], 'same')
            
            h_fc = tf.layers.conv2d(h_conv6, class_count, 1)
            
            # 原始logit输出 无pooling
            h_fc_masked = tf.multiply(h_fc, real_label_mask)
            h_fc_masked = tf.reshape(h_fc_masked, [m * n, class_count])
            
            h_fc_maxpool = h_fc
            
            # 加权损失
            h_fc_pool = tf.nn.softmax(h_fc_maxpool, -1) 
            h_fc_pool_reshape = tf.reshape(h_fc_pool, [m * n, class_count])
            real_labels = tf.reshape(real_label, [m * n, class_count])
            ##  动态权重
            we = -tf.multiply(real_labels,
                              tf.log(h_fc_pool_reshape))
            
            Original_Loss=tf.reduce_sum(we)/tf.reduce_sum(real_labels)#用于计算原始的loss
            
            we = tf.multiply(we, tf.reshape(real_label_mask, [m * n, class_count]))
            # # #固定权重
            we2 = tf.reduce_sum(real_labels, 0) + 1  # 每类训练样本个数 加1是为了防止除0
            we2 = 1. / (we2)  # 每类样本的权重
            we2 = tf.expand_dims(we2, 0)
            we2 = tf.tile(we2, [m * n, 1])
            we = tf.multiply(we, we2)
            pool_cross_entropy = tf.reduce_sum(we)            
            return h_fc_pool, h_fc_masked, pool_cross_entropy, input_img, real_label, real_label_mask, drop,Original_Loss

        #############将train 和 test 和val 样本标签转化为向量形式###################
        # 训练集
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        train_samples_gt_vector = np.zeros([m * n, class_count], np.float)
        for i in range(train_samples_gt.shape[0]):
            class_idx = train_samples_gt[i]
            if class_idx != 0:
                temp = np.zeros([class_count])
                temp[int(class_idx - 1)] = 1
                train_samples_gt_vector[i] = temp
        train_samples_gt_vector = np.reshape(train_samples_gt_vector, [m, n, class_count])
        # 测试集
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        test_samples_gt_vector = np.zeros([m * n, class_count], np.float)
        for i in range(test_samples_gt.shape[0]):
            class_idx = test_samples_gt[i]
            if class_idx != 0:
                temp = np.zeros([class_count])
                temp[int(class_idx - 1)] = 1
                test_samples_gt_vector[i] = temp
        test_samples_gt_vector = np.reshape(test_samples_gt_vector, [m, n, class_count])
        # 验证集
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        val_samples_gt_vector = np.zeros([m * n, class_count], np.float)
        for i in range(val_samples_gt.shape[0]):
            class_idx = val_samples_gt[i]
            if class_idx != 0:
                temp = np.zeros([class_count])
                temp[int(class_idx - 1)] = 1
                val_samples_gt_vector[i] = temp
        val_samples_gt_vector = np.reshape(val_samples_gt_vector, [m, n, class_count])
        
        ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
        # 训练集
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m, n, class_count])
        
        # 测试集
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m, n, class_count])
        
        # 验证集
        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m, n, class_count])
        
        # 构建模型
        h_fc, h_fc_masked, cross_entropy, input_img, real_label, real_label_mask, drop,Original_Loss = model(150, 50, data, gt)
        
        # 训练 使用标签样本光谱信息
        train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99).minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(h_fc_masked, 1), tf.argmax(tf.reshape(real_label, [m * n, class_count]), 1))
        assert correct_prediction.get_shape() == [m * n]
        
        # 计算预测正确的样本个数——计算图
        correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        
        # 将数据扩展一维，以满足网络输入需求
        data = np.reshape(data, [1, m, n, d])
        train_samples_gt_vector = np.expand_dims(train_samples_gt_vector, 0)
        train_label_mask = np.expand_dims(train_label_mask, 0)
        val_samples_gt_vector = np.expand_dims(val_samples_gt_vector, 0)
        
        test_samples_gt_vector = np.expand_dims(test_samples_gt_vector, 0)
        test_label_mask = np.expand_dims(test_label_mask, 0)
        val_label_mask = np.expand_dims(val_label_mask, 0)
        
        zero_vector = np.zeros([class_count])
        all_label_mask = np.ones([1, m, n, class_count])  # 设置一个全1的mask，使得网络输出所有分类标签
        
        my_mask = test_label_mask[0]
        my_mask = np.reshape(my_mask, [m * n, class_count])
        my_mask = my_mask.T
        
        # 输出各集合数量
        print('train set:', len(train_data_index))
        print('val set:', len(val_data_index))
        print('test set:', len(test_data_index))
        
        # 保存模型类
        saver = tf.train.Saver()
        
        # 记录模型训练中的最佳参数
        best_acc = 0
        best_loss = 1e9
        
        # 配置运行环境
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu90%的显存
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        
        train_count_perclass = np.zeros([class_count])
        for x in range(len(train_samples_gt)):
            if train_samples_gt[x] != 0:
                train_count_perclass[int(train_samples_gt[x] - 1)] += 1
        print('train count={}'.format(train_count_perclass))
        less_count_class = np.min(train_count_perclass)
        aaa = np.sum(train_count_perclass)
        
        time_train_start = time.clock()
        
        #记录训练与验证损失
        Train_Loss=[]
        Val_Loss=[]
        
        for i in range(max_epoch + 1):
            # 训练一步
            _, train_right_count, all_pixel_pre_labels,train_pixel_loss = sess.run([train_step, correct_count, h_fc,Original_Loss],
                                                                  feed_dict={input_img: data,
                                                                             real_label: train_samples_gt_vector,
                                                                             real_label_mask: train_label_mask, drop: 1})
            
            if i % 10 == 0:
                # print("{}   train loss={}".format(i,train_pixel_loss))
                # 计算训练精度
                train_acc = (train_right_count - len(background_idx) - len(test_data_index) - len(val_data_index)) / (
                    len(train_data_index))
                # 使用验证集进行验证，计算验证精度，与验证损失
                val_right_count, val_loss = sess.run([correct_count, Original_Loss],
                                                     feed_dict={input_img: data, real_label: val_samples_gt_vector,
                                                                real_label_mask: val_label_mask, drop: 1})
                val_acc = (val_right_count - len(background_idx) - len(test_data_index) - len(train_data_index)) / (
                    len(val_data_index))
                # 输出loss、train_acc、val_acc
                # print('step:', i, 'train_acc=', train_acc, 'val_loss=', val_loss, 'val_acc=', val_acc)
                Train_Loss.append(train_pixel_loss)
                Val_Loss.append(val_loss)
                # print("{}   train loss={}, val loss={}".format(i,train_pixel_loss,val_loss))

                if val_loss < best_loss:
                    best_acc = val_acc
                    best_loss = val_loss
                    saver.save(sess, "model\\best_model")
                    # print('save model')
        
        print('best val loss = {}'.format(best_loss))
        
        time_train_end = time.clock()
        print('training complete.Training time=', time_train_end - time_train_start)

        if 'Loss_curve':
            # 设置图例并且设置图例的字体及大小
            font1 = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 15,
                     }
            # 绘制损失函数曲线
            plt.figure(0)
            plt.tick_params(labelsize=12)
            # history_dict = history_fdssc.history
            loss_value = np.array( Train_Loss)[1:]
            val_loss_value =np.array( Val_Loss)[1:]
            epochs = range(1, loss_value.size + 1) * np.array( 10,dtype=np.int)
            plt.plot(epochs, loss_value, "bo", label="Training loss", linewidth=3, markersize=5)
            plt.plot(epochs, val_loss_value, "r", label="Validation loss", linewidth=3)
            plt.xlabel("Iterations", font1)
            plt.ylabel("Loss", font1)
            plt.legend()
            # plt.show()
            plt.savefig('result_SSCDenseNet\\' + dataset_name + '_LOSS' , dpi=400)  # 指定分辨率
            plt.close()

        # 加载模型 及 评测模型
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('model\\best_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint("model\\"))
            
            time_test_start = time.clock()
            test_count_temp, output_data, all_pixel_pre_labels = sess.run([correct_count, h_fc_masked, h_fc],
                                                                          feed_dict={input_img: data,
                                                                                     real_label: test_samples_gt_vector,
                                                                                     real_label_mask: test_label_mask,
                                                                                     drop: 1.0})
            time_test_end = time.clock()
            
            # 计算OA
            test_OA = (test_count_temp - len(background_idx) - len(train_data_index) - len(val_data_index)) / (
                (m * n - len(background_idx) - len(train_data_index) - len(val_data_index)))  # 9330
            
            # 计算AA
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            for z in range(output_data.shape[0]):
                if ~(zero_vector == output_data[z]).all():
                    idx[z] += 1
            idx = idx + train_samples_gt
            count_perclass = np.zeros([class_count])
            correct_perclass = np.zeros([class_count])
            for x in range(len(test_samples_gt)):
                if test_samples_gt[x] != 0:
                    count_perclass[int(test_samples_gt[x] - 1)] += 1
                    if test_samples_gt[x] == idx[x]:
                        correct_perclass[int(test_samples_gt[x] - 1)] += 1
            test_AC_list = correct_perclass / count_perclass
            test_AA = np.average(test_AC_list)
            
            # 计算KPP
            test_pre_label_list = []
            test_real_label_list = []
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            idx = np.reshape(idx, [m, n])
            for ii in range(m):
                for jj in range(n):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(idx[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16), test_real_label_list.astype(np.int16))
            test_kpp = kappa
            
            # 输出
            print("test OA=", test_OA, "AA=", test_AA, 'kpp=', test_kpp)
            print('acc per class:')
            print(test_AC_list)
            
            
            OA_ALL.append(test_OA)
            AA_ALL.append(test_AA)
            KPP_ALL.append(test_kpp)
            AVG_ALL.append(test_AC_list)
            
            
            # 绘制全像素分类标签图（带背景 and 不带背景）
            print('save picture...')
            all_pixel_pre_labels = np.reshape(all_pixel_pre_labels, [m * n, class_count])
            idx = np.argmax(all_pixel_pre_labels, axis=-1)
            idx += 1
            idx = np.reshape(idx, [m, n])
            Draw_Classification_Map(idx, 'result_SSCDenseNet\\' + dataset_name + '_with_background_' + str(train_ratio))
            sio.savemat('result_SSCDenseNet\\' + dataset_name + '_classify_mat.mat', {'data': idx})  # 保存mat格式分类数据
            
            idx = np.reshape(idx, [-1])
            idx[list(background_idx)] = 0
            idx = np.reshape(idx, [m, n])
            Draw_Classification_Map(idx, 'result_SSCDenseNet\\' + dataset_name + '_without_background_' + str(train_ratio))
            
            # 保存数据信息
            f = open('result_SSCDenseNet\\' + dataset_name + '_results.txt', 'a+')
            str_results = '\n======================' \
                          + " learning rate=" + str(learning_rate) \
                          + " epochs=" + str(max_epoch) \
                          + " train ratio=" + str(train_ratio) \
                          + " val ratio=" + str(val_ratio) \
                          + " ======================" \
                          + "\nOA=" + str(test_OA) \
                          + "\nAA=" + str(test_AA) \
                          + '\nkpp=' + str(test_kpp) \
                          + '\ntrain time:' + str(time_train_end - time_train_start) \
                          + '\ntest time:' + str(time_test_end - time_test_start) \
                          + '\nacc per class:' + str(test_AC_list) + "\n"
            f.write(str_results)
            f.close()
            pass
        tf.reset_default_graph()
        sess.close()
        print('complete.')
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    
    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
