# ConvNeXt:
The ConvNeXt class is a class written in Python that implements the ConvNeXt model, which is a convolutional neural network that uses grouped convolutions and layer scaling to improve performance and efficiency. The ConvNeXt class has the following attributes and methods:

- **model_type:** A string that indicates the type of the model, which can be one of 'tiny', 'small', 'base', 'large' or 'xlarge'. Different model types have different depths and projection dimensions.
- **drop_path_rate:** A float that indicates the probability of stochastic depth, i.e., the probability of each convolutional block being dropped. Stochastic depth can improve the generalization and robustness of the model.
- **layer_scale_init_value:** A float that indicates the initial value of layer scaling, i.e., the coefficient that each convolutional block's output is multiplied by. Layer scaling can stabilize the training and convergence of the model.
- **classes:** An integer that indicates the number of classes for the classification task. If include_top is True, the last layer of the model is a fully connected layer that outputs neurons equal to the number of classes.
- **include_top:** A boolean that indicates whether to include the top layer. If True, the last layer of the model is a fully connected layer that outputs neurons equal to the number of classes. If False, the last layer of the model is a global average pooling layer or a global max pooling layer, depending on the pooling parameter.
- **pooling:** A string or None that indicates the pooling method. If include_top is False, the last layer of the model is a pooling layer, depending on the pooling parameter. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **loss_object:** A TensorFlow object that indicates the loss function. The default loss function is categorical cross-entropy.
- **optimizer:** A TensorFlow object that indicates the optimizer. The default optimizer is Adam.
- **param:** A list that stores all the parameters (weights and biases) of the model.
- **km:** An integer that indicates the kernel mode.
- **build(dtype='float32'):** A method that builds the structure of the model. It accepts one parameter dtype, which indicates the data type, defaulting to 'float32'. This method creates all the convolutional blocks, downsampling blocks, normalization layers, pooling layers and fully connected layers required by the model and stores them in their respective attributes.
- **fp(data, p):** A method that performs forward propagation. It accepts two parameters data and p, which indicate the input data and process number respectively. This method passes the input data through all the layers of the model in turn and returns the output data.
- **loss(output, labels, p):** A method that calculates the loss value. It accepts three parameters output, labels and p, which indicate the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between the output data and true labels and returns the loss value.
- **opt(gradient, p):** A method that performs optimization update. It accepts two parameters gradient and p, which indicate the gradient value and process number respectively. This method uses the optimizer to update all parameters of the model according to gradient value and returns updated parameters.

# MobileNet:
The MobileNet class is a Python class that implements the MobileNet model, which is a type of convolutional neural network that uses depthwise separable convolutions and ReLU6 activation to reduce the computational cost and model size. The MobileNet class has the following attributes and methods:

- **alpha**: A float, indicating the width multiplier that controls the number of filters in each layer. A smaller alpha reduces the number of filters and the model size.
- **depth_multiplier**: A float, indicating the depth multiplier that controls the number of depthwise convolution output channels. A smaller depth_multiplier reduces the number of channels and the model size.
- **dropout**: A float, indicating the dropout rate that is applied to the last layer before the classification layer. Dropout can improve the model's generalization and robustness.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A TensorFlow object, indicating the optimizer. The default optimizer is Adam optimizer.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build(dtype='float32')**: A method, used to build the model's structure. It accepts one argument `dtype`, indicating the data type, defaulting to 'float32'. This method creates all the convolution blocks, depthwise convolution blocks, normalization layers, pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# MobileNetV2:
The MobileNetV2 class is a Python class that implements the MobileNetV2 model, which is a type of convolutional neural network that uses inverted residual blocks and linear bottlenecks to improve performance and efficiency. The MobileNetV2 class has the following attributes and methods:

- **alpha**: A float, indicating the width multiplier that controls the number of filters in each layer. A smaller alpha reduces the number of filters and the model size.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A TensorFlow object, indicating the optimizer. The default optimizer is Adam optimizer.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build(dtype='float32')**: A method, used to build the model's structure. It accepts one argument `dtype`, indicating the data type, defaulting to 'float32'. This method creates all the convolution blocks, depthwise convolution blocks, normalization layers, pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# ResNetRS:
The ResNetRS class is a Python class that implements the ResNet-RS model, which is a type of convolutional neural network that uses residual blocks and stochastic depth to achieve state-of-the-art performance on image classification tasks. The ResNet-RS class has the following attributes and methods:

- **bn_momentum**: A float, indicating the momentum for the batch normalization layers.
- **bn_epsilon**: A float, indicating the epsilon for the batch normalization layers.
- **activation**: A string, indicating the activation function for the convolutional layers. The default activation function is ReLU.
- **se_ratio**: A float, indicating the ratio for the Squeeze and Excitation blocks. The default ratio is 0.25.
- **dropout_rate**: A float, indicating the dropout rate for the last layer before the classification layer. Dropout can improve the model's generalization and robustness.
- **drop_connect_rate**: A float, indicating the initial rate for the stochastic depth. Stochastic depth can improve the model's generalization and robustness by randomly dropping out blocks during training.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **block_args**: A list of dictionaries, indicating the arguments for each block group. Each dictionary contains the input filters and the number of repeats for each block group. The default block arguments are based on the model depth.
- **model_name**: A string, indicating the name of the ResNet-RS variant. The name determines the model depth and block arguments.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **include_preprocessing**: a bool, indicating whether to include preprocessing for the input data. If True, the input data will be rescaled and normalized according to ImageNet statistics.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A TensorFlow object, indicating the optimizer. The default optimizer is Adam optimizer.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build(dtype='float32')**: A method, used to build the model's structure. It accepts one argument `dtype`, indicating the data type, defaulting to 'float32'. This method creates all the stem blocks, block groups and head blocks that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.
