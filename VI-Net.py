_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
BN_EPSILON = 0.001
weight_decay=0.0002

class Model(object):
    def __init__(self, is_training=True, batch_size=32, learning_rate=1e-3, num_labels=4):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels
        self._is_training = is_training

    def activation_summary(self, x):
        '''
        x: A Tensor
        Add histogram summary and scalar summary of the sparsity of the tensor
        '''
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        '''
        name: The name of the new variable
        shape: A list of dimensions
        initializer: User Xavier as default.
        is_fc_layer: To create fc layer variable
        Return he created variable
        '''

        ## TODO: to allow different weight decay to fully connected layer and conv layer
        if is_fc_layer is True:
            regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                        regularizer=regularizer)
        return new_variables



    def batch_normalization_layer(self, x, phase_train, affine=True):
        """
        x: 4D input tensor
        phase_train: True if training phase
        scope: variable scope
        affine: affine-transform outputs
        Return: batch-normalized maps
        """
        shape = x.get_shape().as_list()

        beta = tf.Variable(tf.constant(0.0, shape=[shape[-1]]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[shape[-1]]),
                            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, affine)
        return normed


    def conv_bn_relu_layer(self, input_layer, filter_shape, stride, name):
        '''
        input_layer: 4D input tensor
        filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        stride: stride size for conv
        Return: 4D tensor
        '''

        out_channel = filter_shape[-1]
        filter = self.create_variables(name=name, shape=filter_shape)

        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = self.batch_normalization_layer(conv_layer, self._is_training)

        output = tf.nn.relu(bn_layer)
        return output


    def bn_relu_conv_layer(self, input_layer, filter_shape, stride):
        '''
        input_layer: 4D input tensor
        filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        stride: stride size for conv
        Return: 4D tensor
        '''

        in_channel = input_layer.get_shape().as_list()[-1]

        bn_layer = self.batch_normalization_layer(input_layer, self._is_training)
        relu_layer = tf.nn.relu(bn_layer)

        filter = self.create_variables(name='conv', shape=filter_shape)
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        return conv_layer

    def conv(self, input_layer, filter_shape, stride, relu, name):
        filter = self.create_variables(name='conv', shape=filter_shape)
        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        if relu:
            return tf.nn.relu(conv_layer)
        else:
            return conv_layer

    def pool(self, input_layer, name):
        return tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID', name=name)

    def unpool_with_argmax(self, pool, ind, name = None, ksize=[1, 2, 2, 1]):

        """
        pool: max pooled output tensor
        ind: argmax indices
        ksize: ksize is the same as for the pool
        Return: unpooling tensor
        """
        with tf.variable_scope(name):
            input_shape = pool.get_shape().as_list()
            output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

            flat_input_size = np.prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            pool_ = tf.reshape(pool, [flat_input_size])
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
            b = tf.ones_like(ind) * batch_range
            b = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(ind, [flat_input_size, 1])
            ind_ = tf.concat([b, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
            ret = tf.reshape(ret, output_shape)
            return ret
    def get_bilinear_filter(self, filter_shape, upscale_factor):
        ##filter_shape is [width, height, num_in_channels, num_out_channels]
        kernel_size = filter_shape[1]
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)

        bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                               shape=weights.shape)
        return bilinear_weights
    def upsample_layer(self, bottom, name, upscale_factor=2):

        kernel_size = 2*upscale_factor - upscale_factor%2
        stride = upscale_factor
        strides = [1, stride, stride, 1]
        n_channels = bottom.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            # Shape of the bottom tensor
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, n_channels]
            output_shape = tf.stack(new_shape)

            filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

            weights = self.get_bilinear_filter(filter_shape,upscale_factor)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

        return deconv

    def inference(self, x, x2, s, reuse):
        '''
        x: 4D input tensor
        reuse: False to train, True for validation
        Return: the last layer in the network
        '''

        layers = []
        with tf.variable_scope('conv1_1', reuse=reuse):
            conv1_1 = self.conv_bn_relu_layer(x, [3, 3, 3, 64], 1, 'conv1_1')
            self.activation_summary(conv1_1)
            layers.append(conv1_1)

        with tf.variable_scope('conv1_12', reuse=reuse):
            conv1_12 = self.conv_bn_relu_layer(conv1_1, [3, 3, 64, 64], 1, 'conv1_12')
            conv1_12, pool1_12_indices = tf.nn.max_pool_with_argmax(conv1_12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_12')
            self.activation_summary(conv1_12)
            layers.append(conv1_12)

        with tf.variable_scope('conv1_13', reuse=reuse):
            conv1_13 = self.conv_bn_relu_layer(conv1_12, [3, 3, 64, 128], 1, 'conv1_13')
            self.activation_summary(conv1_13)
            layers.append(conv1_13)

        with tf.variable_scope('conv1_14', reuse=reuse):
            conv1_14 = self.conv_bn_relu_layer(conv1_13, [3, 3, 128, 128], 1, 'conv1_14')
            conv1_14, pool1_14_indices = tf.nn.max_pool_with_argmax(conv1_14, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_14')
            self.activation_summary(conv1_14)
            layers.append(conv1_14)

        with tf.variable_scope('conv1_15', reuse=reuse):
            conv1_15 = self.conv_bn_relu_layer(conv1_14, [3, 3, 128, 256], 1, 'conv1_15')
            self.activation_summary(conv1_15)
            layers.append(conv1_15)

        with tf.variable_scope('conv1_16', reuse=reuse):
            conv1_16 = self.conv_bn_relu_layer(conv1_15, [3, 3, 256, 256], 1, 'conv1_16')
            self.activation_summary(conv1_16)
            layers.append(conv1_16)

        with tf.variable_scope('conv1_17', reuse=reuse):
            conv1_17 = self.conv_bn_relu_layer(conv1_16, [3, 3, 256, 256], 1, 'conv1_17')
            conv1_17, pool1_17_indices = tf.nn.max_pool_with_argmax(conv1_17, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_17')
            self.activation_summary(conv1_17)
            layers.append(conv1_17)

        with tf.variable_scope('conv1_18', reuse=reuse):
            conv1_18 = self.conv_bn_relu_layer(conv1_17, [3, 3, 256, 512], 1, 'conv1_18')
            self.activation_summary(conv1_18)
            layers.append(conv1_18)

        with tf.variable_scope('conv1_19', reuse=reuse):
            conv1_19 = self.conv_bn_relu_layer(conv1_18, [3, 3, 512, 512], 1, 'conv1_19')
            self.activation_summary(conv1_19)
            layers.append(conv1_19)

        with tf.variable_scope('conv1_20', reuse=reuse):
            conv1_20 = self.conv_bn_relu_layer(conv1_19, [3, 3, 512, 512], 1, 'conv1_20')
            conv1_20, pool1_20_indices = tf.nn.max_pool_with_argmax(conv1_20, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_20')
            self.activation_summary(conv1_20)
            layers.append(conv1_20)

        with tf.variable_scope('conv1_21', reuse=reuse):
            conv1_21 = self.conv_bn_relu_layer(conv1_20, [3, 3, 512, 512], 1, 'conv1_21')
            self.activation_summary(conv1_21)
            layers.append(conv1_21)

        with tf.variable_scope('conv1_22', reuse=reuse):
            conv1_22 = self.conv_bn_relu_layer(conv1_21, [3, 3, 512, 512], 1, 'conv1_22')
            self.activation_summary(conv1_22)
            layers.append(conv1_22)

        with tf.variable_scope('conv1_23', reuse=reuse):
            conv1_23 = self.conv_bn_relu_layer(conv1_22, [3, 3, 512, 512], 1, 'conv1_23')
            conv1_23, pool1_23_indices = tf.nn.max_pool_with_argmax(conv1_23, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_23')
            self.activation_summary(conv1_23)
            layers.append(conv1_23)


        with tf.variable_scope('conv3_1', reuse=reuse):
            conv3_1 = self.conv_bn_relu_layer(x2, [3, 3, 2, 64], 1, 'conv3_1')
            self.activation_summary(conv3_1)
            layers.append(conv3_1)

        with tf.variable_scope('conv3_12', reuse=reuse):
            conv3_12 = self.conv_bn_relu_layer(conv3_1, [3, 3, 64, 64], 1, 'conv3_12')
            conv3_12, pool3_12_indices = tf.nn.max_pool_with_argmax(conv3_12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_12')
            self.activation_summary(conv3_12)
            layers.append(conv3_12)

        with tf.variable_scope('conv3_13', reuse=reuse):
            conv3_13 = self.conv_bn_relu_layer(conv3_12, [3, 3, 64, 128], 1, 'conv3_13')
            self.activation_summary(conv3_13)
            layers.append(conv3_13)

        with tf.variable_scope('conv3_14', reuse=reuse):
            conv3_14 = self.conv_bn_relu_layer(conv3_13, [3, 3, 128, 128], 1, 'conv3_14')
            conv3_14, pool3_14_indices = tf.nn.max_pool_with_argmax(conv3_14, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_14')
            self.activation_summary(conv3_14)
            layers.append(conv3_14)

        with tf.variable_scope('conv3_15', reuse=reuse):
            conv3_15 = self.conv_bn_relu_layer(conv3_14, [3, 3, 128, 256], 1, 'conv3_15')
            self.activation_summary(conv3_15)
            layers.append(conv3_15)

        with tf.variable_scope('conv3_16', reuse=reuse):
            conv3_16 = self.conv_bn_relu_layer(conv3_15, [3, 3, 256, 256], 1, 'conv3_16')
            self.activation_summary(conv3_16)
            layers.append(conv3_16)

        with tf.variable_scope('conv3_17', reuse=reuse):
            conv3_17 = self.conv_bn_relu_layer(conv3_16, [3, 3, 256, 256], 1, 'conv3_17')
            conv3_17, pool3_17_indices = tf.nn.max_pool_with_argmax(conv3_17, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_17')
            self.activation_summary(conv3_17)
            layers.append(conv3_17)

        with tf.variable_scope('conv3_18', reuse=reuse):
            conv3_18 = self.conv_bn_relu_layer(conv3_17, [3, 3, 256, 512], 1, 'conv3_18')
            self.activation_summary(conv3_18)
            layers.append(conv3_18)

        with tf.variable_scope('conv3_19', reuse=reuse):
            conv3_19 = self.conv_bn_relu_layer(conv3_18, [3, 3, 512, 512], 1, 'conv3_19')
            self.activation_summary(conv3_19)
            layers.append(conv3_19)

        with tf.variable_scope('conv3_20', reuse=reuse):
            conv3_20 = self.conv_bn_relu_layer(conv3_19, [3, 3, 512, 512], 1, 'conv3_20')
            conv3_20, pool3_20_indices = tf.nn.max_pool_with_argmax(conv3_20, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_20')
            self.activation_summary(conv3_20)
            layers.append(conv3_20)

        with tf.variable_scope('conv3_21', reuse=reuse):
            conv3_21 = self.conv_bn_relu_layer(conv3_20, [3, 3, 512, 512], 1, 'conv3_21')
            self.activation_summary(conv3_21)
            layers.append(conv3_21)

        with tf.variable_scope('conv3_22', reuse=reuse):
            conv3_22 = self.conv_bn_relu_layer(conv3_21, [3, 3, 512, 512], 1, 'conv3_22')
            self.activation_summary(conv3_22)
            layers.append(conv3_22)

        with tf.variable_scope('conv3_23', reuse=reuse):
            conv3_23 = self.conv_bn_relu_layer(conv3_22, [3, 3, 512, 512], 1, 'conv3_23')
            conv3_23, pool3_23_indices = tf.nn.max_pool_with_argmax(conv3_23, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_23')
            self.activation_summary(conv3_23)
            layers.append(conv3_23)


        with tf.variable_scope('condition', reuse=reuse):
            res1 = tf.cond(s>0.0, lambda: conv1_23, lambda:conv3_23 )


        with tf.variable_scope('conv1_23_decode', reuse=reuse):
            unpool1_23 = self.upsample_layer(res1, name="unpool_1_23")
            conv1_23_decode = self.conv_bn_relu_layer(unpool1_23, [3, 3, 512, 512], 1, 'conv1_23_decode')
            self.activation_summary(conv1_23_decode)
            layers.append(conv1_23_decode)
            conv1_22_decode = self.conv_bn_relu_layer(conv1_23_decode, [3, 3, 512, 512], 1, 'conv1_22_decode')
            self.activation_summary(conv1_22_decode)
            layers.append(conv1_22_decode)
            conv1_21_decode = self.conv_bn_relu_layer(conv1_22_decode, [3, 3, 512, 512], 1, 'conv1_21_decode')
            self.activation_summary(conv1_21_decode)
            layers.append(conv1_21_decode)

        with tf.variable_scope('conv1_20_decode', reuse=reuse):
            unpool1_20 = self.upsample_layer(conv1_21_decode, name="unpool_1_20")
            conv1_20_decode = self.conv_bn_relu_layer(unpool1_20, [3, 3, 512, 512], 1, 'conv1_20_decode')
            self.activation_summary(conv1_20_decode)
            layers.append(conv1_20_decode)
            conv1_19_decode = self.conv_bn_relu_layer(conv1_20_decode, [3, 3, 512, 512], 1, 'conv1_19_decode')
            self.activation_summary(conv1_19_decode)
            layers.append(conv1_19_decode)
            conv1_18_decode = self.conv_bn_relu_layer(conv1_19_decode, [3, 3, 512, 512], 1, 'conv1_18_decode')
            self.activation_summary(conv1_18_decode)
            layers.append(conv1_18_decode)

        with tf.variable_scope('conv1_17_decode', reuse=reuse):
            unpool1_17 = self.upsample_layer(conv1_18_decode, name="unpool_1_17")
            conv1_17_decode = self.conv_bn_relu_layer(unpool1_17, [3, 3, 512, 256], 1, 'conv1_17_decode')
            self.activation_summary(conv1_17_decode)
            layers.append(conv1_17_decode)
            conv1_16_decode = self.conv_bn_relu_layer(conv1_17_decode, [3, 3, 256, 256], 1, 'conv1_16_decode')
            self.activation_summary(conv1_16_decode)
            layers.append(conv1_16_decode)
            conv1_15_decode = self.conv_bn_relu_layer(conv1_16_decode, [3, 3, 256, 256], 1, 'conv1_15_decode')
            self.activation_summary(conv1_15_decode)
            layers.append(conv1_15_decode)

        with tf.variable_scope('conv1_14_decode', reuse=reuse):
            unpool1_14 = self.upsample_layer(conv1_15_decode, name="unpool_1_14")
            conv1_14_decode = self.conv_bn_relu_layer(unpool1_14, [3, 3, 256, 128], 1, 'conv1_14_decode')
            self.activation_summary(conv1_14_decode)
            layers.append(conv1_14_decode)
            conv1_13_decode = self.conv_bn_relu_layer(conv1_14_decode, [3, 3, 128, 128], 1, 'conv1_13_decode')
            self.activation_summary(conv1_13_decode)
            layers.append(conv1_13_decode)

        with tf.variable_scope('conv1_12_decode', reuse=reuse):
            unpool1_12 = self.upsample_layer(conv1_13_decode, name="unpool_1_12")
            conv1_12_decode = self.conv_bn_relu_layer(unpool1_12, [3, 3, 128, 64], 1, 'conv1_12_decode')
            self.activation_summary(conv1_12_decode)
            layers.append(conv1_12_decode)
            conv1_1_decode = self.conv_bn_relu_layer(conv1_12_decode, [3, 3, 64, 64], 1, 'conv1_1_decode')
            self.activation_summary(conv1_1_decode)
            layers.append(conv1_1_decode)


        with tf.variable_scope('fc_11', reuse=reuse):
            fc_11 = self.conv(conv1_1_decode, [1, 1, 64, self._num_labels], 1, False, 'fc_11')
            self.activation_summary(fc_11)
            layers.append(fc_11)

        return layers[-1]
