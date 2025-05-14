import pathlib,random
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import unit,os

class Data():

    def __init__(self,args):
        self.data_path = args.data_path
        self.train_num = args.train_num
        self.seed = args.seed
        self.data_name = args.data_name
        self.result = args.result
        self.tfrecords = args.tfrecords
        self.args = args
        self.cube_size = args.cube_size
        
        self.data_dict = sio.loadmat(str(pathlib.Path(self.data_path, self.data_name + '.mat')))
        self.data_gt_dict = sio.loadmat(str(pathlib.Path(self.data_path, self.data_name+'_gt.mat')))
        data_name = [t for t in list(self.data_dict.keys()) if not t.startswith('__')][0]
        data_gt_name = [t for t in list(self.data_gt_dict.keys()) if not t.startswith('__')][0]
        self.data = self.data_dict[data_name]
        self.data = unit.max_min(self.data).astype(np.float32)
        self.data_gt = self.data_gt_dict[data_gt_name].astype(np.int64)
        self.dim = self.data.shape[2]
        print('DataSet %s shape is %s'%(self.data_name,self.data.shape))

    def neighbor_add(self,row, col, w_size=3):  # 给出 row，col和标签，返回w_size大小的cube
        t = w_size // 2
        cube = np.zeros(shape=[w_size, w_size, self.data.shape[2]])
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                if i + row < 0 or i + row >= self.data.shape[0] or j + col < 0 or j + col >= self.data.shape[1]:
                    cube[i + t, j + t] = self.data[row, col]
                else:
                    cube[i + t, j + t] = self.data[i + row, j + col]
        return cube

    def read_data(self):
        data = self.data
        data_gt = self.data_gt
        self.data_gt = data_gt

        sio.savemat(os.path.join(self.result,'info.mat'),{
            'shape':self.data.shape,
            'data':self.data,
            'data_gt':self.data_gt,
            'dim':self.data.shape[2],
            'class_num':np.max(self.data_gt)
        })

        class_num = np.max(data_gt)
        data_pos = {i: [] for i in range(1, class_num + 1)}
        train_pos = {i: [] for i in range(1, class_num + 1)}
        test_pos = {i: [] for i in range(1, class_num + 1)}
        for i in range(data_gt.shape[0]):
            for j in range(data_gt.shape[1]):
                for k in range(1, class_num + 1):
                    if data_gt[i, j] == k:
                        if self.data_name == 'dftc':
                            train_pos[k].append([i, j])
                        else:
                            data_pos[k].append([i, j])
                    if self.data_name == 'dftc':
                        if self.test_gt[i,j]==k:
                            test_pos[k].append([i, j])
        self.data_pos = data_pos
        if self.args.fix_seed:
            random.seed(self.seed)
        
        for k, v in data_pos.items():
            if self.train_num > 0 and self.train_num < 1:
                train_num = self.train_num * len(v)
            elif len(v)<self.train_num:
                train_num = 15
            else:
                train_num = self.train_num
            train_pos[k] = random.sample(v, int(train_num))
            test_pos[k] = [i for i in v if i not in train_pos[k]]
        self.train_pos = train_pos
        self.test_pos = test_pos
        train_pos_all = list()
        test_pos_all = list()
        for k,v in self.train_pos.items():
            for t in v:
                train_pos_all.append([k,t])
        for k,v in self.test_pos.items():
            for t in v:
                test_pos_all.append([k,t])
        train_t = 0
        test_t = 0
        for (k1,v1),(k2,v2) in zip(self.train_pos.items(),self.test_pos.items()):
            print('traindata-ID %s: %s; testdata-ID %s: %s'%(k1,len(v1),k2,len(v2)))
            train_t += len(v1)
            test_t += len(v2)
        print('total train %s, total test %s'%(train_t,test_t))
        # for k,v in self.test_pos.items():
        #     print('testdata-ID %s: %s'%(k,len(v)))
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        # train data
        train_data_name = os.path.join(self.tfrecords, 'train_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(train_data_name)
        for i in train_pos_all:
            [r,c] = i[1]
            pixel_t = self.neighbor_add(r,c,w_size=self.cube_size).astype(np.float32).tostring()
            label_t = np.array(np.array(i[0] - 1).astype(np.int64))
            example = tf.train.Example(features=(tf.train.Features(
                feature={
                    'traindata': _bytes_feature(pixel_t),
                    'trainlabel': _int64_feature(label_t)
                }
            )))
            writer.write(example.SerializeToString())
        writer.close()

        # test data
        test_data_name = os.path.join(self.tfrecords, 'test_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(test_data_name)
        for i in test_pos_all:
            [r, c] = i[1]
            pixel_t = self.neighbor_add(r,c,w_size=self.cube_size).astype(np.float32).tostring()
            label_t = np.array(np.array(i[0] - 1).astype(np.int64))
            example = tf.train.Example(features=(tf.train.Features(
                feature={
                    'testdata': _bytes_feature(pixel_t),
                    'testlabel': _int64_feature(label_t)
                }
            )))
            writer.write(example.SerializeToString())
        writer.close()

        # map data
        map_data_name = os.path.join(self.tfrecords, 'map_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(map_data_name)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data_gt[i,j] == 0:
                    continue
                pixel_t = self.neighbor_add(i, j, w_size=self.cube_size).astype(np.float32).tostring()
                pos = [i,j]
                pos = np.asarray(pos,dtype=np.int64).tostring()
                example = tf.train.Example(features=(tf.train.Features(
                    feature={
                        'mapdata': _bytes_feature(pixel_t),
                        'pos': _bytes_feature(pos),
                    }
                )))
                writer.write(example.SerializeToString())
        writer.close()

        # map seg data
        map_data_name = os.path.join(self.tfrecords, 'map_data_seg.tfrecords')
        writer = tf.python_io.TFRecordWriter(map_data_name)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # if data_gt[i,j] == 0:
                #     continue
                pixel_t = self.neighbor_add(i, j, w_size=self.cube_size).astype(np.float32).tostring()
                pos = [i,j]
                pos = np.asarray(pos,dtype=np.int64).tostring()
                example = tf.train.Example(features=(tf.train.Features(
                    feature={
                        'mapdata': _bytes_feature(pixel_t),
                        'pos': _bytes_feature(pos),
                    }
                )))
                writer.write(example.SerializeToString())
        writer.close()


    def data_parse(self,filename,type='train'):
        dataset = tf.data.TFRecordDataset([filename])
        def parser_train(record):
            keys_to_features = {
                'traindata': tf.FixedLenFeature([], tf.string),
                'trainlabel': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            train_data = tf.decode_raw(features['traindata'], tf.float32)
            train_label = tf.cast(features['trainlabel'], tf.int64)
            shape = [self.cube_size,self.cube_size, self.dim]
            train_data = tf.reshape(train_data, shape)
            train_label = tf.reshape(train_label, [1])
            return train_data, train_label
        def parser_test(record):
            keys_to_features = {
                'testdata': tf.FixedLenFeature([], tf.string),
                'testlabel': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            test_data = tf.decode_raw(features['testdata'], tf.float32)
            test_label = tf.cast(features['testlabel'], tf.int64)
            shape = [self.cube_size,self.cube_size, self.dim]
            test_data = tf.reshape(test_data, shape)
            test_label = tf.reshape(test_label, [1])
            return test_data, test_label
        def parser_map(record):
            keys_to_features = {
                'mapdata': tf.FixedLenFeature([], tf.string),
                'pos': tf.FixedLenFeature([], tf.string),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            map_data = tf.decode_raw(features['mapdata'], tf.float32)
            pos = tf.decode_raw(features['pos'], tf.int64)
            shape = [self.cube_size,self.cube_size, self.dim]
            map_data = tf.reshape(map_data, shape)
            pos = tf.reshape(pos,[2])
            return map_data,pos
        def parser_map_seg(record):
            keys_to_features = {
                'mapdata': tf.FixedLenFeature([], tf.string),
                'pos': tf.FixedLenFeature([], tf.string),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            map_data = tf.decode_raw(features['mapdata'], tf.float32)
            pos = tf.decode_raw(features['pos'], tf.int64)
            shape = [self.cube_size,self.cube_size, self.dim]
            map_data = tf.reshape(map_data, shape)
            pos = tf.reshape(pos,[2])
            return map_data,pos

        if type == 'train':
            dataset = dataset.map(parser_train)
            dataset = dataset.shuffle(buffer_size=20000)
            dataset = dataset.batch(self.args.batch_size)
            dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'test':
            dataset = dataset.map(parser_test)
            dataset = dataset.batch(self.args.test_batch)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'map':
            dataset = dataset.map(parser_map).repeat(1)
            dataset = dataset.batch(self.args.test_batch)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'map_seg':
            dataset = dataset.map(parser_map_seg).repeat(1)
            dataset = dataset.batch(self.args.test_batch)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
