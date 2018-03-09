import tensorflow as tf


def encoder_classif(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):

    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=4)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=8)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1

    # Global average pooling
    x = tf.reduce_mean(x6_, [1,2])
    # Last layer 
    x = tf.layers.dense(x, n_classes)

    return x



def encoder_decoder(input_x=None, n_classes=20, weights=None, width=224, height=224, channels=3, training=True):
    # paddign same, filtros mas pequemos.. 

#hacer capas profundas con poca kernels, tambien hacer con dilataicones y con saltos

    layer_index = 0
    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_simple(input_x, 48, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=4)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x4, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x3 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=8)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x3, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x2 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = tf.layers.average_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1
    x1 = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1
    x6_ = tf.layers.average_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), name='pool'+ str(layer_index))
    layer_index = layer_index + 1


    x6_ = concatenation_convs(x6_, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=24)
    layer_index = layer_index + 1
    
    x = deconv2d_bn(x6_, 96, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x1], axis=3)
    x5_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=16)
    layer_index = layer_index + 1

    x = deconv2d_bn(x5_, 128, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x2], axis=3)
    x4_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=12)
    layer_index = layer_index + 1
    x = deconv2d_bn(x4_, 164, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x3], axis=3)
    x3_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=8)
    layer_index = layer_index + 1
    x = deconv2d_bn(x3_, 196, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x = tf.concat([x, x4], axis=3)
    x2_ = concatenation_convs(x, 16, 3, 3, padding='same', strides=(1, 1), training=training, layer_index=layer_index, times=4)
    layer_index = layer_index + 1
    x = deconv2d_bn(x2_, 224, 3, 3, padding='same', strides=(2, 2), training=training, layer_index=layer_index)
    layer_index = layer_index + 1




    x3_ = deconv2d_bn(x3_, 64, 3, 3, padding='same', strides=(4 ,4), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x4_ = deconv2d_bn(x4_, 48, 3, 3, padding='same', strides=(8, 8), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x5_ = deconv2d_bn(x5_, 32, 3, 3, padding='same', strides=(16, 16), training=training, layer_index=layer_index)
    layer_index = layer_index + 1
    x6_ = deconv2d_bn(x6_, 16, 3, 3, padding='same', strides=(32,32), training=training, layer_index=layer_index)
    layer_index = layer_index + 1


    x = tf.concat([x, x3_, x4_, x5_, x6_], axis=3)
    print(x.get_shape())

    layer_index = layer_index + 1
    x = conv2d_simple(x, n_classes, 1, 1, padding='same', strides=(1, 1), training=training,layer_index=layer_index, last=True)

    print(x.get_shape())

    return x











def conv2d_simple(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True, layer_index=0,last =False):

    with tf.variable_scope('conv2d_simple'+str(layer_index)):


        x = tf.layers.conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!
        if not last:
            x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index)) # scale=False,
            x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))

        return x


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=0, dilated=True ):

    with tf.variable_scope('conv2d_bn_'+str(layer_index)):


        # Bottleneck
        x = tf.layers.conv2d(x, filters*4, (1, 1), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='convbottle'+ str(layer_index)) # there is also dilation_rate!



        if dilated:
            filers_a = filters - int(filters/6) - int(filters/8)
            filers_c = int(filters/8)
            filers_b = int(filters/6)

            x1 = tf.layers.conv2d(x, filers_a, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!

            x2 = tf.layers.conv2d(x, filers_c, (5,5), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv2'+ str(layer_index)) # there is also dilation_rate!

            x3 = tf.layers.conv2d(x, filers_b, (num_row, num_col), padding=padding, dilation_rate=(5,5),activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv3'+ str(layer_index)) # there is also dilation_rate!
            x = tf.concat([x1, x2, x3], axis=3)
        else:
            x = tf.layers.conv2d(x, filters, (num_row, num_col), strides=strides, padding=padding, dilation_rate=dilation_rate,activation=None,   bias_initializer=None,
             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None, name='conv1'+ str(layer_index)) # there is also dilation_rate!


        x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))
       #  x = tf.nn.dropout(x, 0.25)
# tf.contrib.layers.l2_regularizer( scale=0.08)
        return x





def deconv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), training=True, activation=None, layer_index=0):
    with tf.variable_scope('deconv2d_bn_'+str(layer_index)):
        print('deconv')
        print(x.get_shape())
        x = tf.layers.conv2d_transpose(x, filters, (num_row, num_col), strides=strides, padding=padding, name='deconv'+ str(layer_index), bias_initializer=None,
         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), kernel_regularizer=None) # there is also dilation_rate!
        x = tf.layers.batch_normalization(x,  training=training, name='BN'+ str(layer_index)) # scale=False,
        x = tf.nn.leaky_relu(x, name='Lrelu'+ str(layer_index))

        return x

def concatenation_convs(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=(1, 1), training=True,  layer_index=0, times=6, compression=0.5):
    with tf.variable_scope('concat_'+str(layer_index)):

        next_input = x
        for time in xrange(times):

            output = conv2d_bn(next_input, filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training, layer_index=layer_index+time)

            next_input = tf.concat([output, next_input], axis=3)


        if compression:
            filters_pre = next_input.get_shape()[3].value
            compresion_filters = int(compression * filters_pre)
            next_input = conv2d_bn(next_input, compresion_filters, num_row, num_col, padding=padding, strides=strides, dilation_rate=dilation_rate, training=training, layer_index=layer_index+time+1)

        print(next_input.get_shape())

    return next_input


