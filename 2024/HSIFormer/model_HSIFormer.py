import keras
from keras.layers import Conv2D, Conv3D, Reshape

import tensorflow as tf
import tensorflow_addons as tfa


from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    CompatibleExtractPatches,
    conv2d_no_bias,
    drop_block,
    layer_norm,
    mlp_block,
    output_block,
    add_pre_post_process,
)



class MultiHeadRelativePositionalKernelBias(tf.keras.layers.Layer): 
    def __init__(self, input_height=-1, is_heads_first=False, **kwargs):
        super().__init__(**kwargs)
        self.input_height, self.is_heads_first = input_height, is_heads_first

    def build(self, input_shape):
        # input (is_heads_first=False): `[batch, height * width, num_heads, ..., size * size]`
        # input (is_heads_first=True): `[batch, num_heads, height * width, ..., size * size]`
        blocks, num_heads = (input_shape[2], input_shape[1]) if self.is_heads_first else (input_shape[1], input_shape[2])
        size = int(tf.math.sqrt(float(input_shape[-1])))
        height = self.input_height if self.input_height > 0 else int(tf.math.sqrt(float(blocks)))
        width = blocks // height
        pos_size = 2 * size - 1
        initializer = tf.initializers.truncated_normal(stddev=0.02)
        self.pos_bias = self.add_weight(name="positional_embedding", shape=(num_heads, pos_size * pos_size), initializer=initializer, trainable=True)

        idx_hh, idx_ww = tf.range(0, size), tf.range(0, size)
        coords = tf.reshape(tf.expand_dims(idx_hh, -1) * pos_size + idx_ww, [-1])
        bias_hh = tf.concat([idx_hh[: size // 2], tf.repeat(idx_hh[size // 2], height - size + 1), idx_hh[size // 2 + 1 :]], axis=-1)
        bias_ww = tf.concat([idx_ww[: size // 2], tf.repeat(idx_ww[size // 2], width - size + 1), idx_ww[size // 2 + 1 :]], axis=-1)
        bias_hw = tf.expand_dims(bias_hh, -1) * pos_size + bias_ww
        bias_coords = tf.expand_dims(bias_hw, -1) + coords
        bias_coords = tf.reshape(bias_coords, [-1, size**2])[::-1]  # torch.flip(bias_coords, [0])

        bias_coords_shape = [bias_coords.shape[0]] + [1] * (len(input_shape) - 4) + [bias_coords.shape[1]]
        self.bias_coords = tf.reshape(bias_coords, bias_coords_shape)  # [height * width, 1 * n, size * size]
        if not self.is_heads_first:
            self.transpose_perm = [1, 0] + list(range(2, len(input_shape) - 1))  # transpose [num_heads, height * width] -> [height * width, num_heads]

    def call(self, inputs):
        if self.is_heads_first:
            return inputs + tf.gather(self.pos_bias, self.bias_coords, axis=-1)
        else:
            return inputs + tf.transpose(tf.gather(self.pos_bias, self.bias_coords, axis=-1), self.transpose_perm)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"input_height": self.input_height, "is_heads_first": self.is_heads_first})
        return base_config




def LWA(
    inputs, kernel_size=7, num_heads=4, key_dim=0, out_weight=True, qkv_bias=True, out_bias=True, attn_dropout=0, output_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / (float(key_dim) ** 0.5)
    out_shape = cc
    qkv_out = num_heads * key_dim

    should_pad_hh, should_pad_ww = max(0, kernel_size - hh), max(0, kernel_size - ww)
    if should_pad_hh or should_pad_ww:
        inputs = tf.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])
        _, hh, ww, cc = inputs.shape

    qkv = keras.layers.Dense(qkv_out * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    query, key_value = tf.split(qkv, [qkv_out, qkv_out * 2], axis=-1)  # Matching weights from PyTorch
    query = tf.expand_dims(tf.reshape(query, [-1, hh * ww, num_heads, key_dim]), -2)  # [batch, hh * ww, num_heads, 1, key_dim]

    # key_value: [batch, height // kernel_size, width // kernel_size, kernel_size, kernel_size, key + value]
    key_value = CompatibleExtractPatches(sizes=kernel_size, strides=1, padding="VALID", compressed=False)(key_value)
    padded = (kernel_size - 1) // 2
    # torch.pad 'replicate'
    key_value = tf.concat([tf.repeat(key_value[:, :1], padded, axis=1), key_value, tf.repeat(key_value[:, -1:], padded, axis=1)], axis=1)
    key_value = tf.concat([tf.repeat(key_value[:, :, :1], padded, axis=2), key_value, tf.repeat(key_value[:, :, -1:], padded, axis=2)], axis=2)

    key_value = tf.reshape(key_value, [-1, kernel_size * kernel_size, key_value.shape[-1]])
    key, value = tf.split(key_value, 2, axis=-1)  # [batch * block_height * block_width, kernel_size * kernel_size, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch * hh*ww, num_heads, key_dim, kernel_size * kernel_size]
    key = tf.reshape(key, [-1, hh * ww, num_heads, key_dim, kernel_size * kernel_size])  # [batch, hh*ww, num_heads, key_dim, kernel_size * kernel_size]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 1, 3])
    value = tf.reshape(value, [-1, hh * ww, num_heads, kernel_size * kernel_size, key_dim])  # [batch, hh*ww, num_heads, kernel_size * kernel_size, key_dim]
    # print(f">>>> {query.shape = }, {key.shape = }, {value.shape = }")

    # [batch, hh * ww, num_heads, 1, kernel_size * kernel_size]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale
    attention_scores = MultiHeadRelativePositionalKernelBias(input_height=hh, name=name and name + "pos")(attention_scores)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores

    # attention_output = [batch, block_height * block_width, num_heads, 1, key_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.reshape(attention_output, [-1, hh, ww, num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if should_pad_hh or should_pad_ww:
        attention_output = attention_output[:, : hh - should_pad_hh, : ww - should_pad_ww, :]

    if out_weight:
        # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output = keras.layers.Dropout(output_dropout, name=name and name + "out_drop")(attention_output) if output_dropout > 0 else attention_output
    return attention_output


def LWA_block(inputs, attn_kernel_size=7, num_heads=4, mlp_ratio=4, mlp_drop_rate=0, attn_drop_rate=0, drop_rate=0, layer_scale=-1, name=None):
    input_channel = inputs.shape[-1]

    attn = layer_norm(inputs, name=name + "attn_")
    attn = LWA(attn, attn_kernel_size, num_heads, attn_dropout=attn_drop_rate, name=name + "attn_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = keras.layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = layer_norm(attn_out, name=name + "mlp_")
    mlp = mlp_block(mlp, int(input_channel * mlp_ratio), activation="gelu", name=name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return keras.layers.Add(name=name + "output")([attn_out, mlp])

def FExtractor(inputs):

    x = Conv3D(filters=16, kernel_size=(1, 1, 7), activation='relu', padding='same')(inputs)
    x = Conv3D(filters=32, kernel_size=(3, 3, 5), activation='relu',padding='same')(x)
    x = Conv3D(filters=64, kernel_size=(5, 5, 7), activation='relu',padding='same')(x)
    x_shape = x.shape
    x = Reshape((x_shape[1], x_shape[2], x_shape[3]*x_shape[4]))(x)
    x = Conv2D(filters=12, kernel_size=(3,3), activation='relu',padding='same')(x)

    return x


def HSIFormer(
    num_blocks=[3, 4],
    out_channels=[64, 128],
    num_heads=[2, 2],
    stem_width=-1,
    attn_kernel_size=7,
    mlp_ratio=3,
    layer_scale=-1,
    input_shape=(12, 12, 12,1),
    num_classes=5,
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="PolF",
    kwargs=None,
):
    """ConvTokenizer stem"""
    inputs = keras.layers.Input(input_shape)
    x=FExtractor(inputs)

    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = conv2d_no_bias(x, stem_width // 2, kernel_size=3, strides=2, use_bias=True, padding="SAME", name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width, kernel_size=3, strides=2, use_bias=True, padding="SAME", name="stem_2_")
    nn = layer_norm(nn, name="stem_")

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, num_head) in enumerate(zip(num_blocks, out_channels, num_heads)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            ds_name = stack_name + "downsample_"
            nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=2, padding="SAME", name=ds_name)
            nn = layer_norm(nn, name=ds_name)
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = LWA_block(nn, attn_kernel_size, num_head, mlp_ratio, drop_rate=block_drop_rate, layer_scale=layer_scale, name=block_name)
            global_block_id += 1
    nn = layer_norm(nn, name="pre_output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    weight_decay = 0.0001
    
    learning_rate = 0.0005
    optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
    
 '''
    model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                
            ],
        )
    '''
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    

    return model






