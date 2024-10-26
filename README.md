# Introduction:
Deep learning models built with Note are compatible with TensorFlow and can be trained with TensorFlow. The documentation shows how to train, test, save, and restore models built with Note.


# Train:
```python
import tensorflow as tf
from Note.models.docs_example.DL.model1 import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model=Model()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

model.train(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If use early stopping.
# model.end_acc=0.9
# model.train(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If save the model at intervals of 1 epoch, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=1
# model. max_save_files=2
# model.train(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If save the model at intervals of 1875 batch, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq_=1875
# model. max_save_files=2
# model.train(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If save parameters only
# model.path='param.dat'
# model.save_freq=1
# model. max_save_files=2
# model.save_param_only=True
# model.train(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.monitor='val_loss'
# model.train(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If set steps_per_execution
# model.path='model.dat'
# model.end_acc=0.9
# model.steps_per_execution=1875
# model.train(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If use parallel test(experiment)
# x_test, y_test = model.segment_data(x_test, y_test, 7)
# test_ds = [tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32) x_test,y_test for zip(x_test,y_test)]
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# test_loss = [tf.keras.metrics.Mean(name='test_loss') for _ in range(7)]
# test_accuracy = [tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy') for _ in range(7)]
# model.train(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy, 7, parallel_test=True)

# visualize
# model.visualize_train()
# model.visualize_test()
# model.visualize_comparison()

# save
# model.save_param('param.dat')
# model.save('model.dat')
```


# Distributed training:
## MirroredStrategy:
```python
import tensorflow as tf
from Note.models.docs_example.DL.model2 import Model

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE) 

with strategy.scope():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)

with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')

with strategy.scope():
  model=Model()
  optimizer = tf.keras.optimizers.Adam()

model.distributed_training(train_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
EPOCHS, train_accuracy=train_accuracy, test_dataset=test_dataset, test_loss=test_loss, test_accuracy=test_accuracy)

# If use early stopping.
# model.end_acc=0.9
# model.distributed_training(train_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy=train_accuracy, test_dataset=test_dataset, test_loss=test_loss, test_accuracy=test_accuracy)

# If save the model at intervals of 2 epoch, with a maximum of 3 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=2
# model.max_save_files=3
# model.distributed_training(train_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy=train_accuracy, test_dataset=test_dataset, test_loss=test_loss, test_accuracy=test_accuracy)

# If save the model at intervals of 1094 batch, with a maximum of 3 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq_=1094
# model.max_save_files=3
# model.distributed_training(train_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy=train_accuracy, test_dataset=test_dataset, test_loss=test_loss, test_accuracy=test_accuracy)

# If save parameters only
# model.path='param.dat'
# model.save_freq=2
# model.max_save_files=3
# model.save_param_only=True
# model.distributed_training(train_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy=train_accuracy, test_dataset=test_dataset, test_loss=test_loss, test_accuracy=test_accuracy)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.monitor='val_loss'
# model.distributed_training(train_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy=train_accuracy, test_dataset=test_dataset, test_loss=test_loss, test_accuracy=test_accuracy)

# If set steps_per_execution
# model.path='model.dat'
# model.end_acc=0.9
# model.steps_per_execution=1094
# model.distributed_training(train_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy=train_accuracy, test_dataset=test_dataset, test_loss=test_loss, test_accuracy=test_accuracy)

# visualize
# model.visualize_train()
# model.visualize_test()
# model.visualize_comparison()

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
## MultiWorkerMirroredStrategy:
```python
import tensorflow as tf
from Note.models.docs_example.DL.model2 import Model
import numpy as np
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')

def mnist_dataset():
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # You need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000)
  return train_dataset

train_dataset = mnist_dataset()

tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
  # Model building needs to be within `strategy.scope()`.
  multi_worker_model = Model()
  # The creation of optimizer and train_accuracy needs to be in
  # `strategy.scope()` as well, since they create variables.
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')

per_worker_batch_size = 64
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers

multi_worker_model.distributed_training(train_dataset, loss_object, global_batch_size, optimizer, strategy,
num_epochs=3, num_steps_per_epoch=70, train_accuracy=train_accuracy)

# If use early stopping.
# model.end_acc=0.9
# multi_worker_model.distributed_training(train_dataset, loss_object, global_batch_size, optimizer, strategy,
# num_epochs=3, num_steps_per_epoch=70, train_accuracy=train_accuracy)

# If save the model at intervals of 2 epoch, with a maximum of 3 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=2
# model.max_save_files=3
# multi_worker_model.distributed_training(train_dataset, loss_object, global_batch_size, optimizer, strategy,
# num_epochs=3, num_steps_per_epoch=70, train_accuracy=train_accuracy)

# If save the model at intervals of 70 batch, with a maximum of 3 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq_=70
# model.max_save_files=3
# multi_worker_model.distributed_training(train_dataset, loss_object, global_batch_size, optimizer, strategy,
# num_epochs=3, num_steps_per_epoch=70, train_accuracy=train_accuracy)

# If save parameters only
# model.path='param.dat'
# model.save_freq=2
# model.max_save_files=3
# model.save_param_only=True
# multi_worker_model.distributed_training(train_dataset, loss_object, global_batch_size, optimizer, strategy,
# num_epochs=3, num_steps_per_epoch=70, train_accuracy=train_accuracy)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.monitor='val_loss'
# multi_worker_model.distributed_training(train_dataset, loss_object, global_batch_size, optimizer, strategy,
# num_epochs=3, num_steps_per_epoch=70, train_accuracy=train_accuracy)

# If set steps_per_execution
# model.path='model.dat'
# model.end_acc=0.9
# model.steps_per_execution=70
# multi_worker_model.distributed_training(train_dataset, loss_object, global_batch_size, optimizer, strategy,
# num_epochs=3, num_steps_per_epoch=70, train_accuracy=train_accuracy)

# visualize
# model.visualize_train()
# model.visualize_test()
# model.visualize_comparison()

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
## ParameterServerStrategy:
```python
import multiprocessing
import os
import portpicker
import tensorflow as tf

def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=i,
        config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=i,
        protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)
variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=NUM_PS))

strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)

def dataset_fn():
# Define dataset_fn.

def test_dataset_fn():
# Define test_dataset_fn.

with strategy.scope():
  # Create the model. The input needs to be compatible with Keras processing layers.
  model = 
  optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.1)
  loss_object = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
  accuracy = tf.keras.metrics.Accuracy()

model.distributed_training(loss_object=loss_object, optimizer=optimizer, strategy=strategy,
  num_epochs=7, num_steps_per_epoch=7, train_accuracy=accuracy, dataset_fn=dataset_fn, test_dataset_fn=test_dataset_fn, eval_steps_per_epoch=7)
```


# Fine-tuning:
```python
model.fine_tuning(10,flag=0)
optimizer.lr=0.0001
fine_ds = tf.data.Dataset.from_tensor_slices((x_fine, y_fine)).batch(32)

EPOCHS = 1

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()

  for images, labels in fine_ds:
    train_step(images, labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
  )
```
flag=0:
Replace the pre-trained layer and assign the parameters of the fine-tuning layer to self.param.

flag=1:
Assign the parameters of the pre-trained layer and the parameters of the fine-tuning layer to self.param.

flag=2:
Restore the pre-trained layer and assign the parameters of the pre-trained layer to self.param.


# Use neural network:
```python
model.training()
output=model(data)
```


# Test model:
```python
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_loss, test_acc = model.test(test_ds, loss_object, test_loss, test_accuracy)
```
or parallel test
```python
import multiprocessing as mp
x_test, y_test = model.segment_data(x_test, y_test, 7)
test_ds = [tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32) x_test,y_test for zip(x_test,y_test)]
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
test_loss = [tf.keras.metrics.Mean(name='test_loss') for _ in range(7)]
test_accuracy = [tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy') for _ in range(7)]
test_loss, test_acc = model.test(test_ds, loss_object, test_loss, test_accuracy, 7, mp)
```


# Save model parameters:
```python
import pickle
output_file=open('param.dat','wb')
pickle.dump(model.param,output_file)
output_file.close()
```
or
```python
model.save_param('param.dat')
```


# Restore model parameters:
```python
import pickle
input_file=open('param.dat','rb')
param=pickle.load(input_file)
input_file.close()
```
or
```python
model.restore_param('param.dat')
```
or
```python
from Note import nn
param=nn.restore_param('param.dat')
```


# Assign the trained parameters to the model:
The assign_param function allows you to assign trained parameters, such as downloaded pre-trained parameters, to the parameters of a neural network. These parameters should be stored in a list.
```python
from Note import nn
from Note.models.tf.ConViT import convit_tiny
import pickle
model=convit_tiny(embed_dim=48)
input_file=open('param.dat','rb')
param=pickle.load(input_file)
input_file.close()
nn.assign_param(model.param,param)
```


# Save model:
```python
import pickle
output_file=open('model.dat','wb')
pickle.dump(model,output_file)
output_file.close()
```
or
```python
model.save('model.dat')
```


# Restore model:
```python
import pickle
input_file=open('model.dat','rb')
model=pickle.load(input_file)
input_file.close()
```
or
```python
model.restore('model.dat')
```
or
```python
from Note import nn
model,optimizer=nn.restore('model.dat')
```


# Build models:
## ConvNeXt_tiny
```python
from Note.models.tf.ConvNeXt import ConvNeXt
convnext_tiny=ConvNeXt(model_type='tiny',classes=1000)
```

## ConvNeXtV2_atto
```python
from Note.models.tf.ConvNeXtV2 import ConvNeXtV2
convnext_atto=ConvNeXtV2(model_type='atto',classes=1000)
```

## CLIP_large
```python
from Note.models.tf.CLIP import CLIP
clip=CLIP(
    embed_dim=1024,
    image_resolution=224,
    vision_layers=14,
    vision_width=1024,
    vision_patch_size=32,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12
  )
```

## DiT_B_4
```python
from Note.models.tf.DiT import DiT_B_4
dit=DiT_B_4()
```

## EfficientNetB0
```python
from Note.models.tf.EfficientNet import EfficientNet
efficientnetb0=EfficientNet(model_name='B0',classes=1000)
```

## EfficientNetV2S
```python
from Note.models.tf.EfficientNetV2 import EfficientNetV2
efficientnetv2s=EfficientNetV2(model_name='efficientnetv2-s',classes=1000)
```

## Llama2_7B
```python
from Note.models.tf.Llama2 import Llama2
llama=Llama2()
```

## MobileNetV2
```python
from Note.models.tf.MobileNetV2 import MobileNetV2
mobilenet=MobileNetV2(classes=1000)
```

## MobileNetV3_large
```python
from Note.models.tf.MobileNetV3 import MobileNetV3
mobilenet=MobileNetV3(model_type="large",classes=1000)
```

## ResNet50
```python
from Note.models.tf.ResNet.ResNet50 import ResNet50
resnet50=ResNet50(classes=1000)
```

## ViT
```python
from Note.models.tf.ViT import ViT
vit=ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    pool='cls',
    channels=3,
    dim_head=64,
    drop_rate=0.1,
    emb_dropout=0.1
)
```

## CaiT
```python
import tensorflow as tf
from Note.models.tf.CaiT import cait_XXS24_224

model = cait_XXS24_224()

img = tf.random.normal((1, 224, 224, 3))

output = model(img) # (1, 1000)
```

## PiT
```python
import tensorflow as tf
from Note.models.tf.PiT import pit_b

model = pit_b()

# forward pass now returns predictions and the attention maps

img = tf.random.normal((1, 224, 224, 3))

output = model(img) # (1, 1000)
```

## Cross ViT
```python
import tensorflow as tf
from Note.models.tf.CrossViT import crossvit_tiny_224()

model = crossvit_tiny_224()

img = tf.random.normal((1, 240, 240, 3))

output = model(img) # (1, 1000)
```

## Deep ViT
```python
import tensorflow as tf
from Note.models.tf.DeepViT import DeepViT

v = DeepViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout_rate = 0.1,
    emb_dropout = 0.1
)

img = tf.random.normal((1, 256, 256, 3))

output = v(img) # (1, 1000)
```

## ViViT
```python
import tensorflow as tf
from Note.models.tf.ViViT import ViViT

v = ViViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    spatial_depth = 6,         # depth of the spatial transformer
    temporal_depth = 6,        # depth of the temporal transformer
    heads = 8,
    mlp_dim = 2048
)

video = tf.random.normal((4, 16, 128, 128, 3)) # (batch, frames, height, width, channels)

output = v(video) # (4, 1000)
```

## XCiT
```python
import tensorflow as tf
from Note.models.tf.XCiT import xcit_nano_12_p16

model = xcit_nano_12_p16()

img = tf.random.normal([1, 224, 224, 3])

output = model(img) # (1, 1000)
```

## CvT
```python
import tensorflow as tf
from Note.models.tf.CvT import CvT

v = CvT(
    num_classes = 1000,
    s1_emb_dim = 64,        # stage 1 - dimension
    s1_emb_kernel = 7,      # stage 1 - conv kernel
    s1_emb_stride = 4,      # stage 1 - conv stride
    s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
    s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
    s1_heads = 1,           # stage 1 - heads
    s1_depth = 1,           # stage 1 - depth
    s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
    s2_emb_dim = 192,       # stage 2 - (same as above)
    s2_emb_kernel = 3,
    s2_emb_stride = 2,
    s2_proj_kernel = 3,
    s2_kv_proj_stride = 2,
    s2_heads = 3,
    s2_depth = 2,
    s2_mlp_mult = 4,
    s3_emb_dim = 384,       # stage 3 - (same as above)
    s3_emb_kernel = 3,
    s3_emb_stride = 2,
    s3_proj_kernel = 3,
    s3_kv_proj_stride = 2,
    s3_heads = 4,
    s3_depth = 10,
    s3_mlp_mult = 4,
    dropout = 0.
)

img = tf.random.normal((1, 224, 224, 3))

output = v(img) # (1, 1000)
```

## CCT
```python
import tensorflow as tf
from Note.models.tf.CCT import CCT

cct = CCT(
    img_size = (224, 448),
    embedding_dim = 384,
    n_conv_layers = 2,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_layers = 14,
    num_heads = 6,
    mlp_ratio = 3.,
    num_classes = 1000,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)

img = tf.random.normal((1, 224, 448, 3))
output = cct(img) # (1, 1000)
```
Alternatively you can use one of several pre-defined models [2,4,6,7,8,14,16] which pre-define the number of layers, number of attention heads, the mlp ratio, and the embedding dimension.
```python
from Note.models.tf.CCT import cct_14

cct = cct_14(
    img_size = 224,
    n_conv_layers = 1,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_classes = 1000,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)
```

## MiT
```python
import tensorflow as tf
from Note.models.tf.MiT import mit_b0
model = mit_b0()

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size, in_chans])

output = model(img)
```

## BEiT
```python
import tensorflow as tf
from Note.models.tf.BEiT import beit_base_patch16_224
model = beit_base_patch16_224()

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size,in_chans])

output = model(img)
```

## SwinMLP
```python
import tensorflow as tf
from Note.models.tf.SwinMLP import SwinMLP
model = SwinMLP()

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size,in_chans])

output = model(img)
```

## SwinTransformerV2
```python
import tensorflow as tf
from Note.models.tf.SwinTransformerV2 import SwinTransformerV2
model = SwinTransformerV2()

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size,in_chans])

output = model(img)
```

## ConViT
```python
import tensorflow as tf
from Note.models.tf.ConViT import convit_tiny
model = convit_tiny(embed_dim=48)

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size,in_chans])

output = model(img)
```

## PVT
```python
import tensorflow as tf
from Note.models.tf.PVT import pvt_v2_b0

model = pvt_v2_b0()

img = tf.random.normal([1, 224, 224, 3])

output = model(img) # (1, 1000)
```

## GCViT
```python
import tensorflow as tf
from Note.models.tf.GCViT import gc_vit_xxtiny

model = gc_vit_xxtiny()

img = tf.random.normal([1, 224, 224, 3])

output = model(img) # (1, 1000)
```

## DaViT
```python
import tensorflow as tf
from Note.models.tf.DaViT import davit_tiny

model = davit_tiny()

img = tf.random.normal([1, 224, 224, 3])

output = model(img) # (1, 1000)
```

# Model's functions
These functions extend the Model class, allowing you to manage namespaces for layers, control freezing and unfreezing of layers, and set training or evaluation modes. Additionally, functions can be applied to layers for initialization or configuration. Below are the descriptions and usage of each function.

**Example**:
```python
from Note import nn

class Block:
  def __init__(self):
    nn.Model.add()
    nn.Model.namespace('block')
    self.layer1 = nn.dense(7, 7)
    self.layer2 = nn.dense(7, 7)
    nn.Model.namespace()
    nn.Model.apply(self.init_weights)
    
    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.trunc_normal_(l.weight, std=0.2))

    def __call__(self, x):
      return self.layer2(self.layer1)

class Model:
  def __init__(self):
    self.block=Block()

  def __call__(self, x):
    return self.block(x)

model = Model()
```

---

## 1. **`add()`**
   - **Function**: Adds a new layer name to the model and tracks the layers added sequentially.
   - **Effect**: Increments the `Model.counter` by 1 and appends a new layer name to `Model.name_list` as `'layer' + str(Model.counter)`.

   **Result**: Adds a new layer to `Model.name_list`, and the layer will be named `'layer1'`, `'layer2'`, and so on. 

   **Relation to `apply()`**: `add()` is typically called before `apply()`. It registers a new layer in the model, and then `apply()` can be used to initialize or modify that layer's parameters.

---

## 2. **`apply(func)`**
   - **Function**: Applies a given function `func` to each layer in the current namespace or initializes layer weights with `func`.
   - **Parameters**:
     - `func` (`callable`): A function to apply to each layer. If a layer has an `input_size`, the function is applied immediately. Otherwise, it assigns `func` to `layer.init_weights`.
   - **Effect**: It iterates through the layers in `Model.layer_dict` under the current `Model.name_`, applies the function to layers with an `input_size`, or initializes layers by assigning the function to their `init_weights`.

   **Result**: The `init_weights` function is applied to layers that have an `input_size`. Layers without an `input_size` will have their `init_weights` attribute set to the `init_weights` function.

   **Relation to `add()`**: After calling `add()` to register a layer, `apply()` can then be used to apply transformations or initialize the layer’s weights. This ensures that operations are performed on all relevant layers in the model.

---

## 3. **`training(self, flag=False)`**
   - **Function**: Sets the entire model or individual layers to training or evaluation mode.
   - **Parameters**:
     - `flag` (`bool`, optional): 
       - `False` (default): Sets the model to evaluation mode.
       - `True`: Sets the model to training mode.
   - **Effect**: Updates the `train_flag` attribute of all layers in `self.layer_list`. If a layer does not have a `train_flag` attribute, it uses the `training` attribute instead. 

   **Example**:
   ```python
   model.training(flag=True)
   ```
   **Result**: Sets all layers in the model to training mode by adjusting either `train_flag` or `training` attributes.

---

## 4. **`namespace(name=None)`**
   - **Function**: Assigns a namespace to layers in the model for tracking layers and parameters.
   - **Parameters**: 
     - `name` (`str`, optional): The name for the namespace of the model. If `None` is passed, no name is assigned to the model.
   - **Effect**: This function adds the layer name to `Model.name_list_`.

   **Result**: The namespace for the model is set to `block`.

---

## 5. **`freeze(self, name=None)`**
   - **Function**: Freezes the parameters of the model or a specific namespace, making them untrainable during training.
   - **Parameters**:
     - `name` (`str`, optional): Specifies the namespace to freeze. If `name` is `None`, it freezes the parameters in all namespaces.
   - **Effect**: This function iterates through all parameters in `self.layer_param` and sets them to be untrainable (`_trainable=False`).

   **Example**:
   ```python
   model.freeze('block')
   ```
   **Result**: Freezes all layer parameters in the `block` namespace, preventing them from being updated during training.

---

## 6. **`unfreeze(self, name=None)`**
   - **Function**: Unfreezes the parameters of the model or a specific namespace, making them trainable again.
   - **Parameters**:
     - `name` (`str`, optional): Specifies the namespace to unfreeze. If `name` is `None`, it unfreezes the parameters in all namespaces.
   - **Effect**: Iterates through all parameters in `self.layer_param` and sets them to be trainable (`_trainable=True`).

   **Example**:
   ```python
   model.unfreeze('block')
   ```
   **Result**: Unfreezes all layer parameters in the `block` namespace, allowing them to be updated during training.

---

## 7. **`eval(self, name=None, flag=True)`**
   - **Function**: Sets the model or specific namespaces to training or evaluation mode.
   - **Parameters**:
     - `name` (`str`, optional): Specifies the namespace to configure. If `name` is `None`, it iterates through all namespaces.
     - `flag` (`bool`, optional): 
       - `True`: Sets to evaluation mode (freezes layers).
       - `False`: Sets to training mode.
   - **Effect**: Controls the training state of each layer. When `flag=True`, the model is set to evaluation mode, and `train_flag=False`.

   **Example**:
   ```python
   model.eval('block', flag=True)
   ```
   **Result**: Sets all layers in `block` to evaluation mode (`train_flag=False`).

---

## Typical Use Cases:

- **Adding layers**:
  - `add()` helps to keep track of the layers as they are added to the model, ensuring unique names are assigned sequentially.
- **Applying functions to layers**:
  - Use `apply()` to apply initialization or transformation functions to model layers, useful for weight initialization or custom configuration of layers after they have been added by `add()`.
- **Global training or evaluation mode**:
  - Use `training()` to set the entire model to training or evaluation mode. This is useful for switching between modes before starting the training or inference processes.
- **Naming layers in the model**: 
  - When you want to control different blocks independently, use `namespace()` to assign a unique name to different layers or modules.
- **Freezing or unfreezing layers**:
  - Use `freeze()` and `unfreeze()` to control which layers participate in gradient updates during training. For example, when fine-tuning a model, you may only want to unfreeze the top layers.
- **Setting training or evaluation modes**:
  - `eval()` allows you to easily switch between training and evaluation modes. During training, you may need to freeze certain layers or switch behaviors in some layers (like Batch Normalization, which behaves differently during training and inference).

These methods provide flexibility in managing complex models, particularly when freezing parameters, applying functions, and adjusting training strategies.

---

## 8. **`train`**

This method implements the training loop for the model, handling both training and testing (optional) over multiple epochs. It allows for configurable options like JIT (Just-In-Time) compilation, parallel testing, and automatic saving of model parameters.

---

### **Parameters**:

- **`train_ds`** (`Dataset`): 
  - The dataset used for training. It should provide pairs of training data and corresponding labels.
  
- **`loss_object`** (`Loss Function`): 
  - The loss function used to compute the training loss for each batch.

- **`train_loss`** (`Metric`): 
  - The metric that tracks the loss over the training batches.

- **`optimizer`** (`Optimizer`): 
  - The optimizer that updates model parameters based on computed gradients.

- **`epochs`** (`int`, optional): 
  - Number of epochs for which the model should be trained. If `None`, training will run indefinitely until manually stopped.

- **`train_accuracy`** (`Metric`, optional): 
  - The metric to track the accuracy during training. If not provided, accuracy tracking is skipped.

- **`test_ds`** (`Dataset`, optional): 
  - The dataset used for testing/validation during training. If `None`, no testing is performed.

- **`test_loss`** (`Metric`, optional): 
  - The metric that tracks the test loss after each epoch.

- **`test_accuracy`** (`Metric`, optional): 
  - The metric to track the test accuracy after each epoch. If not provided, accuracy tracking for the test dataset is skipped.

- **`processes`** (`int`, optional): 
  - The number of processes used for parallel testing. It is only used if `parallel_test` is set to `True`.

- **`parallel_test`** (`bool`, optional): 
  - If `True`, testing will be performed in parallel using multiprocessing. Otherwise, testing is done sequentially.

- **`jit_compile`** (`bool`, optional, default=`True`): 
  - If `True`, Just-In-Time (JIT) compilation is enabled for faster execution of the training step.

- **`p`** (`int`, optional): 
  - Controls the frequency of printing metrics during training. If `None`, it defaults to 9, and the printing frequency is derived from the number of epochs.

---

### **Behavior**:

1. **Epoch and Printing Frequency**:
   - The frequency of printing epoch metrics is determined by parameter `p`. If `p` is not provided, it is computed from the total number of epochs to ensure metrics are printed at regular intervals.
   - If `parallel_test=True`, multiprocessing will be enabled for testing the model after each epoch.

2. **Training Loop**:
   - For each epoch, the model iterates over the training dataset `train_ds`.
   - If `jit_compile=True`, the `train_step` function will be JIT compiled for faster execution. Otherwise, the uncompiled version `train_step_` will be used.
   - During training, after every `steps_per_execution` steps (if defined), the model will compute the current training loss and accuracy. Testing will also be performed periodically if a `test_ds` is provided.

3. **Testing**:
   - After each epoch or at defined intervals, the model is tested on the `test_ds` (if provided), and the test loss and accuracy are tracked.
   - Results from both training and testing are stored in corresponding lists (`train_loss_list`, `train_acc_list`, `test_loss_list`, and `test_acc_list`).

4. **Model Saving**:
   - The model will save its parameters at defined intervals based on `save_freq_` or after certain steps.
   - Two saving modes are supported: saving the entire model or saving only the parameters, controlled by `save_param_only`.

5. **Metrics Logging**:
   - The training loss is logged at the end of each epoch, along with the training accuracy (if applicable).
   - If a test dataset is provided, the test loss and accuracy are also printed at regular intervals.

6. **Time Tracking**:
   - The total training time for each epoch is tracked, and the cumulative training time is printed at the end of the training process.

7. **Termination**:
   - The training process terminates based on the `epochs` parameter or a custom stopping criterion defined by `self.end()`.

---

### **Return**:
This method returns `None`. However, it logs the training and testing metrics and saves the model/parameters at intervals. It also prints the total time taken for training.

---

### **Example Usage**:

```python
# Assuming train_ds, test_ds, loss_object, optimizer are defined

model.train(
    train_ds=train_ds, 
    loss_object=loss_object, 
    train_loss=train_loss, 
    optimizer=optimizer, 
    epochs=50, 
    train_accuracy=train_accuracy, 
    test_ds=test_ds, 
    test_loss=test_loss, 
    test_accuracy=test_accuracy,
    processes=4,
    parallel_test=True,
    jit_compile=True
)
```

---

This method provides flexibility in model training, especially for large models where parallel testing or periodic saving is required.

---

## 9. **`distributed_training`**

### **Description:**
The `distributed_training` function is responsible for performing distributed training across different TensorFlow distributed strategies, including `MirroredStrategy`, `MultiWorkerMirroredStrategy`, and `ParameterServerStrategy`. It allows training to be scaled across multiple devices (GPUs, TPUs, or across multiple machines), handling both training and evaluation logic with support for different dataset distributions, batch processing, and optimization across the distributed system.

### **Parameters:**

- **`train_dataset`** (optional, `tf.data.Dataset`):  
   The training dataset used for distributed training.

- **`loss_object`** (optional, `tf.keras.losses.Loss`):  
   The loss function used to compute the model's error during training.

- **`global_batch_size`** (optional, `int`):  
   The global batch size used for distributed training, accounting for all replicas involved in the process.

- **`optimizer`** (optional, `tf.keras.optimizers.Optimizer`):  
   The optimizer used to update the model's weights.

- **`strategy`** (`tf.distribute.Strategy`):  
   The distributed strategy (e.g., `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `ParameterServerStrategy`) that defines how training is distributed across multiple devices.

- **`epochs`** (optional, `int`):  
   The total number of epochs for training, specifically used with `MirroredStrategy`.

- **`num_epochs`** (optional, `int`):  
   The total number of epochs for training, particularly used with `MultiWorkerMirroredStrategy` and `ParameterServerStrategy`.

- **`num_steps_per_epoch`** (optional, `int`):  
   The number of steps to execute in each epoch during training.

- **`train_accuracy`** (optional, `tf.keras.metrics.Accuracy`):  
   A metric to track the model's accuracy on the training dataset.

- **`test_dataset`** (optional, `tf.data.Dataset`):  
   The test dataset used for evaluation during training.

- **`test_loss`** (optional, `tf.keras.metrics.Mean`):  
   A metric to track the loss on the test dataset.

- **`test_accuracy`** (optional, `tf.keras.metrics.Accuracy`):  
   A metric to track accuracy on the test dataset.

- **`dataset_fn`** (optional, `function`):  
   A function to preprocess or create the training dataset, mainly for use with `MultiWorkerMirroredStrategy` or `ParameterServerStrategy`.

- **`test_dataset_fn`** (optional, `function`):  
   A function to create the test dataset, particularly for `ParameterServerStrategy`.

- **`global_test_batch_size`** (optional, `int`):  
   The global batch size for test data evaluation.

- **`eval_steps_per_epoch`** (optional, `int`):  
   The number of evaluation steps to run after each epoch during test set evaluation.

- **`jit_compile`** (`bool`, default `True`):  
   Whether to enable TensorFlow's JIT (Just-In-Time) compilation to optimize training execution speed.

- **`p`** (optional, `int`):  
   A parameter used for adjusting the printing frequency of training progress during epochs. If not provided, defaults to `9`.

### **Returns:**
None. The function prints loss, accuracy, and other statistics during training and saves the model at regular intervals, if specified.

---

### **Key Features:**

1. **Flexible Epoch Management:**  
   The function allows for flexible epoch management with both `epochs` (for `MirroredStrategy`) and `num_epochs` (for `MultiWorkerMirroredStrategy` and `ParameterServerStrategy`). It also manages how frequently training progress should be printed, based on the total number of epochs and the value of `p`.

2. **Distributed Strategy Handling:**  
   The function supports three main distributed strategies:
   - **`MirroredStrategy`**: Used for synchronous training across multiple GPUs on a single machine, utilizing `epochs`.
   - **`MultiWorkerMirroredStrategy`**: For distributed training across multiple workers, distributing datasets and training workloads, utilizing `num_epochs`.
   - **`ParameterServerStrategy`**: For training across multiple machines with parameter servers coordinating distributed computation, also utilizing `num_epochs`.

3. **Training and Evaluation Loop:**  
   The function alternates between training steps and evaluation steps (if `test_dataset` is provided), computing the loss and accuracy for both datasets. It also allows Just-In-Time (JIT) compilation to optimize training performance.

4. **Automatic Model Saving:**  
   The function provides options to save the model parameters or the entire model at regular intervals, based on the training progress.

5. **JIT Compilation Option:**  
   `jit_compile=True` enables TensorFlow's JIT compilation, which can improve training performance by compiling parts of the computation graph into optimized code.

6. **Custom Dataset Function:**  
   For strategies like `MultiWorkerMirroredStrategy` and `ParameterServerStrategy`, the function supports creating datasets dynamically using `dataset_fn` and `test_dataset_fn`.

### **Usage Example:**
```python
# Example usage for distributed training with MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = Model()
model.distributed_training(
    train_dataset=train_dataset,
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(),
    global_batch_size=64,
    optimizer=tf.keras.optimizers.Adam(),
    strategy=strategy,
    epochs=10,  # Used for MirroredStrategy
    train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(),
    test_dataset=test_dataset,
    test_loss=tf.keras.metrics.Mean(),
    test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
)

# Example usage for distributed training with MultiWorkerMirroredStrategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
  model = Model()
model.distributed_training(
    train_dataset=train_dataset,
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(),
    global_batch_size=64,
    optimizer=tf.keras.optimizers.Adam(),
    strategy=strategy,
    num_epochs=10,  # Used for MultiWorkerMirroredStrategy or ParameterServerStrategy
    train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(),
    test_dataset=test_dataset,
    test_loss=tf.keras.metrics.Mean(),
    test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
)
```

This documentation provides a detailed overview of the function, its parameters, and how it interacts with TensorFlow's distributed training strategies.

---

# **Building a Neural Network by Inheriting from the `Model` Class**

## **Define a Custom Model Class**

1. Create a custom model class that inherits from `nn.Model`.
2. Define network layers in the `__init__` method.
3. Set up the `__call__` method to define how input data propagates through the network.

Example code:

```python
from Note import nn

class Model(nn.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.conv2d(32, 3, activation='relu')  # Convolutional layer with 32 filters and ReLU activation
        self.flatten = nn.flatten()                       # Flatten layer to reshape the output from the convolutional layer
        self.d1 = nn.dense(128, activation='relu')        # Dense layer with 128 units and ReLU activation
        self.d2 = nn.dense(10)                            # Output layer with 10 units (for classifying the 10 MNIST digits)

    def __call__(self, x):
        x = self.conv1(x)    # Apply convolutional layer
        x = self.flatten(x)  # Apply flatten layer
        x = self.d1(x)       # Apply first dense layer
        return self.d2(x)    # Output layer (logits without activation)
```

## **Code Explanation**

1. **Convolutional Layer (`self.conv1`)**: Defines a 2D convolutional layer to extract features from the input images.
2. **Flatten Layer (`self.flatten`)**: Reshapes the convolutional output into a one-dimensional vector to feed into dense layers.
3. **Dense Layer (`self.d1`)**: Defines a dense layer with 128 units for high-level feature extraction.
4. **Output Layer (`self.d2`)**: Defines an output layer with 10 units for classifying the 10 digit classes (0-9).

## **Summary**

By inheriting from the `Model` class, you can create a neural network with a custom structure. The `__init__` method is used to define the layers, and the `__call__` method sets up the forward propagation logic. This setup allows for easy addition of new layers and flexible expansion of the network structure.

---

These are the foundational steps for building a neural network by inheriting from the `Model` class.
