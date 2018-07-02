# coding=utf-8
import tensorflow as tf
import tensorlayer as tl

# res-1
def forward(x, is_train, **kwargs):
    '''need: dropout'''
    keep = kwargs['dropout'] if is_train else 1.0
    act_fn = tf.nn.relu

    net = tl.layers.InputLayer(x, name="tlayer_input")
    net = tl.layers.Conv2dLayer(net, shape=[5,5,1,32], name="tlayer_conv_1")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_1")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_1")

    net = tl.layers.Conv2dLayer(net, shape=[5,5,32,64], name="tlayer_conv_2")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_2")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_2")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,64,128], name="tlayer_conv_3")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_3")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_3")

    net = tl.layers.Conv2dLayer(net, shape=[5,5,128,256], name="tlayer_conv_4")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_4")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_4")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,256,384], name="tlayer_conv_5")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_5")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_5")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,384,384], name="tlayer_conv_6")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_6")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,384,384], name="tlayer_conv_7")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_7")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,384,256], name="tlayer_conv_8")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_8")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_8")

    net = tl.layers.FlattenLayer(net, name="tlayer_flatten")

    net = tl.layers.DenseLayer(net, 2048, act=act_fn, name="tlayer_dense_1")
    net = tl.layers.DropoutLayer(net, keep=keep, is_train=is_train, name="tlayer_dense_dropout_1")

    net = tl.layers.DenseLayer(net, 1024, name="tlayer_dense_2")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_dense_bn_2")
    net = tl.layers.DropoutLayer(net, keep=keep, is_train=is_train, name="tlayer_dense_bn_dropout_2")

    net = tl.layers.DenseLayer(net, 512, name="tlayer_dense_3")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_dense_bn_3")
    net = tl.layers.DropoutLayer(net, keep=keep, is_train=is_train, name="tlayer_dense_bn_dropout_3")

    net = tl.layers.DenseLayer(net, 1, act=tf.nn.sigmoid, name="tlayer_dense_4")

    return net
# res-3
def forward3(x, is_train, **kwargs):


    keep = kwargs['dropout'] if is_train else 1.0
    act_fn = tf.nn.relu

    net = tl.layers.InputLayer(x, name="tlayer_input")
    net = tl.layers.Conv2dLayer(net, shape=[5,5,1,32], name="tlayer_conv_1")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_1")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_1")

    net = tl.layers.Conv2dLayer(net, shape=[5,5,32,64], name="tlayer_conv_2")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_2")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_2")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,64,128], name="tlayer_conv_3")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_3")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_3")

    net = tl.layers.Conv2dLayer(net, shape=[5,5,128,256], name="tlayer_conv_4")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_4")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_4")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,256,384], name="tlayer_conv_5")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_5")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_5")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,384,384], name="tlayer_conv_6")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_6")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,384,384], name="tlayer_conv_7")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_7")

    net = tl.layers.Conv2dLayer(net, shape=[3,3,384,256], name="tlayer_conv_8")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_conv_bn_8")
    net = tl.layers.PoolLayer(net, ksize=[1,3,3,1],  padding='VALID', name="tlayer_conv_bn_pool_8")

    net = tl.layers.FlattenLayer(net, name="tlayer_flatten")

    # net = tl.layers.DenseLayer(net, 2048, act=act_fn, name="tlayer_dense_1")
    # net = tl.layers.DropoutLayer(net, keep=keep, is_train=is_train, name="tlayer_dense_dropout_1")

    net = tl.layers.DenseLayer(net, 1024, name="tlayer_dense_2")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_dense_bn_2")
    net = tl.layers.DropoutLayer(net, keep=keep, is_train=is_train, name="tlayer_dense_bn_dropout_2")

    net = tl.layers.DenseLayer(net, 512, name="tlayer_dense_3")
    net = tl.layers.BatchNormLayer(net, act=act_fn, is_train=is_train, name="tlayer_dense_bn_3")
    net = tl.layers.DropoutLayer(net, keep=keep, is_train=is_train, name="tlayer_dense_bn_dropout_3")

    net = tl.layers.DenseLayer(net, 1, act=tf.nn.sigmoid, name="tlayer_dense_4")

    return net


def model2(is_train, **kwargs):
    '''
    kwargs:
        lr: float, fixed learning rate.
        dropout: float, if is_train = True, dropout should be provided.
        show_layers_info: bool, whethere to print layers information of tensorlayers.
    '''
    show_layers_info = kwargs['show_layers_info']
    if is_train:
        dropout = kwargs['dropout']
        learning_rate = kwargs['lr']
    else:
        dropout = 1.0

    x = tf.placeholder(tf.float32, [None, 447, 447, 1], name="input_X")
    if show_layers_info:
        net = forward(x, is_train=is_train, dropout=dropout)
    else:
        with tl.ops.suppress_stdout():
            net = forward(x, is_train=is_train, dropout=dropout)
    y_pred = net.outputs

    if is_train:
        # y_real = tf.placeholder(tf.int32, [None,], name="input_y")
        y_real = tf.placeholder(tf.float32, [None,], name="input_y")
        y_real_ = tf.divide(tf.reshape(y_real, [-1, 1]), 100)
        loss = tf.losses.mean_squared_error(y_real_, y_pred, scope="loss")
        gstep = tf.Variable(0, trainable=False)
        lr = tf.Variable(learning_rate, trainable=False)
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss, gstep)
        tf.summary.scalar("mse", loss)
        tf.summary.scalar("lr", lr)
        return {'train_op': train_op, 'loss': loss, 'gstep': gstep, 'lr': lr,
                'prediction': y_pred, 'net': net, 'y_real': y_real, 'x': x}
    else:
        return {'prediction': y_pred, 'net': net, 'x': x}
# for res-1.1
def model(is_train, **kwargs):
    '''
    kwargs:
        lr: float, fixed learning rate.
        dropout: float, if is_train = True, dropout should be provided.
        show_layers_info: bool, whethere to print layers information of tensorlayers.
    '''
    show_layers_info = kwargs['show_layers_info']
    if is_train:
        dropout = kwargs['dropout']
        learning_rate = kwargs['lr']
    else:
        dropout = 1.0

    x = tf.placeholder(tf.float32, [None, 447, 447, 1], name="input_X")
    if show_layers_info:
        net = forward(x, is_train=is_train, dropout=dropout)
    else:
        with tl.ops.suppress_stdout():
            net = forward(x, is_train=is_train, dropout=dropout)
    y_pred = net.outputs

    if is_train:
        y_real = tf.placeholder(tf.float32, [None,], name="input_y")
        y_real_ = tf.divide(tf.reshape(y_real, [-1, 1]), 100)
        # weight design
        weights = tf.abs(y_real_ - 0.2) + 0.1
        loss = tf.losses.mean_squared_error(y_real_, y_pred, weights= weights, scope="loss")
        gstep = tf.Variable(0, trainable=False)
        lr = tf.Variable(learning_rate, trainable=False)
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss, gstep)
        tf.summary.scalar("mse", loss)
        tf.summary.scalar("lr", lr)
        return {'train_op': train_op, 'loss': loss, 'gstep': gstep, 'lr': lr,
                'prediction': y_pred, 'net': net, 'y_real': y_real, 'x': x}
    else:
        return {'prediction': y_pred, 'net': net, 'x': x}


def read_tfr(tfr_fps, std_shape=(447, 447, 1)):
    filename_queue = tf.train.string_input_producer(tfr_fps)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.float32),
            'image' : tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, std_shape)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.float32)
    return [img, label]


if __name__ == '__main__':
    x = tf.ones([16,447,447,1],dtype=tf.float32)
    # with tl.ops.suppress_stdout():
    net = forward(x, True, dropout=0.5)
    # print(555)
    # print(net[0])