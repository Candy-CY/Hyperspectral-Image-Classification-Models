def spectral_attention(input_fea):


    """
    输入：特征图
    输出：加权特征图, 光谱权值

    双尺度注意力模块
    通过考虑光谱局部通道和跨通道交互来获取更具综合性的权值

    Input: feature maps
    Output: spectral-weighted feature maps, spectral weight(later used for interleave-attention)
    Dual channel attention module
    More comprehensive weights are obtained by considering the local channel and cross-channel interaction of the spectral

    """
    
    cross_k = 3  # the range of cross-channel interaction

    h = int(input_fea.shape[1])
    print('input_fea.shape_size[1]:', h)
    w = int(input_fea.shape[2])
    print('input_fea.shape_size[2]:', w)

    av_pool = AveragePooling3D((h, w, 1))(input_fea)    

    conv_3 = Convolution3D(filters=24, kernel_size=(1, 1, cross_k), padding='same', activation='sigmoid')(av_pool)
    conv_1 = Convolution3D(filters=24, kernel_size=(1, 1, 1), padding='same', activation='sigmoid')(av_pool)
  
    conv = add([conv_1, conv_3])   

    spe_weights = UpSampling3D((h, w, 1))(conv)  
    mul = merge.Multiply()([input_fea, spe_weights]) 
    bn = BatchNormalization()(mul)

    return bn, spe_weights

def spatial_attention(input_fea):

    '''
    输入：特征图
    输出：加权特征图, 返回权值

    空间注意力：
    使用1*1*1卷积层直接获取全局平均和最大值平均下的空间权重

    Input: feature maps
    Output: spatial-weighted feature maps, return spatial weight(later used for interleave-attention)
    Spatial attention:
    Use the 1 * 1 * 1 convolution layer to directly obtain the spatial weight under the average-pooling and max-pooling average

    '''

    c = int(input_fea.shape[3])
    print('channel number is ', c)

    aver = AveragePooling3D((1, 1, c))(input_fea)   
    maxp = MaxPooling3D((1, 1, c))(input_fea)

    conv1 = Conv3D(filters=24, kernel_size=(1, 1, 1), activation='sigmoid')(aver)
    conv2 = Conv3D(filters=24, kernel_size=(1, 1, 1), activation='sigmoid')(maxp)

    fuse = add([conv1, conv2])                 

    spa_weights = UpSampling3D((1, 1, c))(fuse)         
    mul = merge.Multiply()([input_fea, spa_weights])    
    bn = BatchNormalization()(mul)

    return bn, spa_weights

def spatial_feature_extraction(input_fea):

    """
    输入：特征图
    输出：深层特征图, 返回权值

    提取空间特征：
    进行两次空间特征提取，并随后进行空间注意力加权

    spatial Branch

    Input: feature maps
    Output: deep spatial feature maps, return spatial weight(later used for interleave-attention)
    Extract spatial features:
    Spatial feature extraction is carried out twice, and then spatial attention weighting is carried out
    """

    conv1 = Conv3D(filters=24, kernel_size=(7, 7, 1), padding='same', activation='relu')(input_fea) 
    conv2 = Conv3D(filters=24, kernel_size=(7, 7, 1), padding='same', activation='relu')(conv1)
    bn = BatchNormalization()(conv2)

    spa_a, spatial_weight = spatial_attention(bn)
    bn_1 = BatchNormalization()(spa_a)

    return bn_1, spatial_weight

def spectral_feature_extraction(input_fea):

    """
    输入：特征图
    输出：深层特征图

    提取光谱特征：
    进行两次光谱特征提取，并随后进行光谱注意力加权

    Spectral branch

    Input: feature maps
    Output: deep spectral feature maps
    Extract spectral features:
    The spectral features were extracted twice, and then the spectral attention was weighted

    """


    conv1 = Conv3D(filters=24, kernel_size=(1, 1, 7), padding='same', activation='relu')(input_fea) 
    conv2 = Conv3D(filters=24, kernel_size=(1, 1, 7), padding='same', activation='relu')(conv1)
    bn = BatchNormalization()(conv2)

    spe_a, spe_weight = spectral_attention(bn)    
    bn_1 = BatchNormalization()(spe_a)

    return bn_1, spe_weight

def spatial_spectral_extraction(input_fea, spa_weight, spe_weight, kernel_num = 24, k_size = (3, 3, 3)):

    """
    输入：特征图、空间注意力权值、光谱注意力权值、卷积核数量、卷积核尺寸
    输出：深层特征图

    提取光谱空间特征：
    进行两次光谱空间特征提取，并进行光谱空间注意力加权；光谱、空间注意力权值均来自于对应支路

    Hybrid branch

    Input: feature maps, spatial attention weight, spectral attention weight, number of convolution kernels, size of convolution kernels
    Output: deep spectral-spatial feature maps
    Extract spectral-spatial joint features:
    Two times spectral-spatial feature extraction, spectral and spatial attention weighting; Spectral and spatial attention weights are from the spectral and spatial  branches

    """


    k = kernel_num
    s = k_size

    conv1 = Conv3D(filters=k, kernel_size=s, padding='same', activation='relu')(input_fea)
    conv2 = Conv3D(filters=k, kernel_size=s, padding='same', activation='relu')(conv1)

    bn = BatchNormalization()(conv2)

    spe_a = merge.Multiply()([bn, spe_weight]) 
    bn_1 = BatchNormalization()(spe_a)

    spa_a =  merge.Multiply()([bn, spa_weight]) 
    bn_2 = BatchNormalization()(spa_a)

    bn_3 = add([bn_1, bn_2])

    return bn_3
    
def Network(input_layer):

    '''
    
    Feature extraction, weighting and interleave-attention mechanism
    
    '''

    # 空间特征提取(spatial feature extraction for the first time)
    spa_fea, spa_weight_1 = spatial_feature_extraction(input_layer)     

    # 光谱特征提取(spectral feature extraction for the first time)
    spe_fea, spe_weight_1 = spectral_feature_extraction(input_layer)


    # 第一次权重交互（空间与光谱域）(interleave-attention: sharing weight to other branches for  the first time)
    spa_fea = merge.Multiply()([spa_fea, spe_weight_1])    # 空间特征加权光谱权重
    spa_fea = BatchNormalization()(spa_fea)

    spe_fea = merge.Multiply()([spe_fea, spa_weight_1])    # 光谱特征加权空间权重
    spe_fea = BatchNormalization()(spe_fea)

    # (feature extraction for the second time)
    spa_fea_1, spa_weight_2 = spatial_feature_extraction(spa_fea)       # 空间特征提取
    spe_fea_1, spe_weight_2 = spectral_feature_extraction(spe_fea)      # 光谱特征提取

    # (interleave-attention: sharing weight to other branches for  the 2nd time)
    spa_fea_1 = merge.Multiply()([spa_fea, spe_weight_2])    # 空间特征加权光谱权重
    spa_fea_1 = BatchNormalization()(spa_fea_1)

    spe_fea_1 = merge.Multiply()([spe_fea, spa_weight_2])    # 光谱特征加权空间权重
    spe_fea_1 = BatchNormalization()(spe_fea_1)    

    # 光谱-空间联合特征提取(Hybrid branch's feature extraction with interleave-attention mechanism)
    spa_spe_fea = spatial_spectral_extraction(input_layer, spa_weight_1, spe_weight_1)
    spa_spe_fea_1 = spatial_spectral_extraction(spa_spe_fea, spa_weight_2, spe_weight_2)

    # 对三通道特征进行融合 (fuse three branch's obtained features)
    fea_spa_spe = add([spe_fea_1, spa_fea_1])
    fea_spa_spe_1 = add([fea_spa_spe, spa_spe_fea_1])
    bn = BatchNormalization()(fea_spa_spe_1)


    h = int(input_layer.shape[1])
    print('input_fea.shape_size[1]:', h)
    w = int(input_layer.shape[2])
    print('input_fea.shape_size[2]:', w)


    pool1 = MaxPooling3D((h, w, 1))(bn)

    f1 = Flatten()(pool1)

    drop1 = Dropout(0.5)(f1)

    dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(drop1)

    model = Model(inputs=input_layer, outputs=dense)
    return model

model = Network(input_layer)
model.summary()
