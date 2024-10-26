# Train:
## Note and Keras:
Agent built with Note or Keras.
```python
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DQN import DQN
# from Note.models.docs_example.RL.keras.DQN import DQN

model=DQN(4,128,2)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(train_loss, optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.train(train_loss, optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.train(train_loss, optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.train(train_loss, optimizer, 100)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# Use PPO.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.PPO import PPO
# from Note.models.docs_example.RL.keras.PPO import PPO

model=PPO(4,128,2,0.7,0.7)
model.set(policy=rl.SoftmaxPolicy(),pool_size=10000,batch=64,update_steps=1000,PPO=True)
optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
```
```python
# Use HER.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DDPG_HER import DDPG
# from Note.models.docs_example.RL.keras.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.MADDPG import DDPG
# from Note.models.docs_example.RL.keras.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(policy=rl.SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MA=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DQN import DQN
# from Note.models.docs_example.RL.keras.pool_network.DQN import DQN

model=DQN(4,128,2,7)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=True, processes=7)
```
```python
# Use HER.
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005,7)
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,trial_count=10,HER=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 2000, pool_network=True, processes=7, processes_her=4)
```
```python
# Use prioritized replay.
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DQN_PR import DQN
# from Note.models.docs_example.RL.keras.pool_network.DQN_PR import DQN

model=DQN(4,128,2,7)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=True, processes=7, processes_pr=4)
```
## PyTorch:
Agent built with PyTorch.
```python
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.DQN import DQN

model=DQN(4,128,2)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(optimizer, 100)

# If use prioritized replay.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# model.train(optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.train(optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.train(optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.train(optimizer, 100)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# Use HER.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(policy=rl.SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MA=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 100)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.pool_network.DQN import DQN

model=DQN(4,128,2,7)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100, pool_network=True, processes=7)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.pool_network.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005,7)
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(train_loss, optimizer, 2000, pool_network=True, processes=7, processes_her=4)
```

# Distributed training:
Agent built with Note or Keras.
## MirroredStrategy:
```python
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DQN import DQN
# from Note.models.docs_example.RL.keras.DQN import DQN

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2)
  optimizer = tf.keras.optimizers.Adam()
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# Use PPO.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.PPO import PPO
# from Note.models.docs_example.RL.keras.PPO import PPO

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=PPO(4,128,2,0.7,0.7)
  optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]

model.set(policy=rl.SoftmaxPolicy(),pool_size=10000,update_steps=1000,PPO=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)
```
```python
# Use HER.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DDPG_HER import DDPG
# from Note.models.docs_example.RL.keras.DDPG_HER import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,criterion=-5,trial_count=10,HER=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 2000)
```
```python
# Use Multi-agent reinforcement learning
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.MADDPG import DDPG
# from Note.models.docs_example.RL.keras.MADDPG import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set(policy=rl.SoftmaxPolicy(),pool_size=3000,trial_count=10,MA=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DQN import DQN
# from Note.models.docs_example.RL.keras.pool_network.DQN import DQN

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2,7)
  optimizer = tf.keras.optimizers.Adam()
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100, pool_network=True, processes=7)
```
```python
# Use HER.
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DDPG_HER import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005,7)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,trial_count=10,HER=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 2000, pool_network=True, processes=7, processes_her=4)
```
## MultiWorkerMirroredStrategy:
```python
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DQN import DQN
# from Note.models.docs_example.RL.keras.pool_network.DQN import DQN
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')

tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

strategy = tf.distribute.MultiWorkerMirroredStrategy()
per_worker_batch_size = 64
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers

with strategy.scope():
  multi_worker_model = DQN(4,128,2)
  optimizer = tf.keras.optimizers.Adam()

multi_worker_model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
                    pool_network=True, processes=7)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```

# `RL.set`:

## **Function Description**:
The `set` function configures various parameters of the Reinforcement Learning (RL) agent. These parameters control the policy, noise, experience pool, batch size, update frequency, and training termination conditions. By adjusting these settings, users can fine-tune the agent's behavior and training process to suit specific RL tasks.

---

## **Function Signature**:
```python
def set(self, 
        policy=None, 
        noise=None, 
        pool_size=None, 
        batch=None, 
        update_batches=None, 
        update_steps=None, 
        trial_count=None, 
        criterion=None, 
        PPO=False, 
        HER=False, 
        MA=False, 
        PR=False, 
        epsilon=None, 
        initial_TD=7., 
        alpha=0.7):
```

---

## **Parameter Description**:

- **`policy`** (`rl.Policy` or `None`):  
  Specifies the policy object for the agent, which controls how actions are selected in each state.

- **`noise`** (`float` or `None`):  
  Adds noise to the action selection process, typically used in continuous action spaces to encourage exploration. The default value is `None`.

- **`pool_size`** (`int` or `None`):  
  The size of the experience replay pool, i.e., the maximum number of experiences that can be stored in the pool. If not set, the pool size defaults to the internal value.

- **`batch`** (`int` or `None`):  
  The size of the batch sampled from the experience pool during training, affecting how much data is used in each training step.

- **`update_batches`** (`int` or `None`):  
  The number of batches to use when updating the network, applicable when using a pool network for storing experiences. Defaults to `None`.

- **`update_steps`** (`int` or `None`):  
  The frequency at which the target Q network is updated, in terms of the number of steps.

- **`trial_count`** (`int` or `None`):  
  Specifies the number of trials over which the average reward is computed during training. After every `trial_count` episodes, the agent's performance is evaluated by calculating the average reward over those episodes. If not set, no average reward is calculated.

- **`criterion`** (`float` or `None`):  
  Specifies the threshold used to terminate training. If `trial_count` is set, the average reward over the most recent `trial_count` episodes is calculated. If this average reward meets or exceeds `criterion`, training is terminated early. This helps avoid unnecessary training once the desired performance level is achieved.

- **`PPO`** (`bool`):  
  Whether to use the Proximal Policy Optimization (PPO) algorithm.

- **`HER`** (`bool`):  
  Whether to use Hindsight Experience Replay (HER), typically used for goal-oriented tasks.

- **`MA`** (`bool`):  
  Whether to use Multi-Agent (MA) reinforcement learning.

- **`PR`** (`bool`):  
  Whether to use Prioritized Experience Replay (PR), a technique to sample experiences based on their significance.

- **`epsilon`** (`float` or `None`):  
  The `ε` value used in an `ε-greedy` policy, controlling the probability of choosing a random action to encourage exploration.

- **`initial_TD`** (`float`):  
  The initial TD-error value used in Prioritized Replay. A higher TD-error leads to higher prioritization of the sample in the experience pool.

- **`alpha`** (`float`):  
  The `α` value used in Prioritized Replay, determining how much the TD-error influences sample prioritization. A higher `α` increases the importance of prioritizing higher TD-error experiences.

---

## **Usage Example**:

```python
# Create an instance of a DQN agent
model = DQN(state_dim=4, hidden_dim=128, action_dim=2)

# Set the agent's policy, experience pool size, batch size, and early stopping conditions
model.set(
    policy=rl.EpsGreedyQPolicy(epsilon=0.01),  # Use epsilon-greedy policy
    pool_size=10000,                           # Set experience pool size
    batch=64,                                  # Set batch size
    update_steps=10,                           # Update target network every 10 steps
    trial_count=100,                           # Calculate average reward every 100 trials
    criterion=200.0,                           # Stop training if average reward reaches 200
    PR=True,                                   # Enable Prioritized Replay
    initial_TD=7.0,                            # Initial TD-error set to 7.0
    alpha=0.7                                  # Alpha value for prioritized sampling
)
```

In this example, the agent computes the average reward every 100 trials. If the average reward reaches 200 or higher, the training process stops early. This method allows the agent to stop training once it reaches a desired performance level, improving training efficiency.

# `RL.train`:
## **Description**:
This function handles the training loop of the reinforcement learning (RL) agent. It supports both single-process and multi-process training, along with the option to use a **pool network** for experience replay. Additionally, it provides support for Hindsight Experience Replay (HER), Prioritized Experience Replay (PR), and optional just-in-time (JIT) compilation for performance optimization.

## **Arguments**:

- **`train_loss`** (`tf.keras.metrics.Metric`): The loss metric used to evaluate the training loss during the optimization process.
  
- **`optimizer`** (`tf.keras.optimizers.Optimizer`): The optimizer used to update the model parameters during training.
  
- **`episodes`** (`int`, optional): The number of training episodes to run. If `None`, the training will continue indefinitely until a stopping criterion is met.

- **`jit_compile`** (`bool`, optional, default=`True`): Whether to enable TensorFlow's JIT compilation for improved performance during training.
  
- **`pool_network`** (`bool`, optional, default=`True`): Whether to use a pool network for experiences collection.
  
- **`processes`** (`int`, optional): Number of parallel processes to use for data collection when using a pool network. If `None`, multi-processing is disabled.
  
- **`processes_her`** (`int`, optional): Number of parallel processes dedicated to Hindsight Experience Replay (HER). Only used if HER is enabled.
  
- **`processes_pr`** (`int`, optional): Number of parallel processes dedicated to Prioritized Experience Replay (PR). Only used if PR is enabled.
  
- **`shuffle`** (`bool`, optional, default=`False`): If `True`, experiences in the pool will be shuffled before sampling. This can help prevent overfitting to recent experiences.
  
- **`p`** (`int`, optional): A parameter that determines the update frequency for logging and printing intermediate results. If `None`, it defaults to `9`.

## **Returns**:
- No return value. The function prints progress at specified intervals and updates the model's parameters based on the training procedure.

## **Details**:
1. **Multiprocessing Setup**:
   - If `pool_network=True`, the function sets up parallel processes to collect experiences in parallel using Python's `multiprocessing` library. Each process collects states, actions, rewards, and other necessary information, which are then aggregated into a shared experience pool.
   
2. **Training Procedure**:
   - If a pool network is used, the agent gathers experiences from multiple parallel environments or processes and stores them in a shared memory pool. The training loop then samples batches from this pool to update the agent's neural network. Otherwise, the agent igathers experiences from environment and stores them in a pool and then updates the network using a different training method (`train2`).
   
   - For each episode, the loss is computed and accumulated in `self.loss_list`. This loss represents the agent's learning progress, and the model parameters are updated using the provided optimizer.
   
3. **Handling Special Experience Replay**:
   - **Hindsight Experience Replay (HER)**: If HER is enabled, the function creates additional processes to manage HER-specific experience sampling and updates.
   
   - **Prioritized Experience Replay (PR)**: If PR is enabled, a prioritized experience replay buffer is updated with the TD-errors (Temporal Difference) of the experiences.

4. **Logging and Saving**:
   - The function prints progress messages every `p` episodes and logs key metrics like average reward and loss. The model can be saved at regular intervals (`self.save_freq`) and upon achieving a certain reward criterion (`self.criterion`).

5. **Termination Criteria**:
   - Training continues until the specified number of episodes (`episodes`) is reached, or in infinite mode (when `episodes=None`), until the reward criterion is met.

## **Usage Example**:

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Start training for 100 episodes using a pool network with 8 processes
agent.train(train_loss=train_loss, optimizer=optimizer, episodes=100, pool_network=True, processes=8)
```

--- 

This documentation provides a detailed explanation of each parameter and the internal behavior of the function, which should be useful for understanding its usage in reinforcement learning training loops.

# `RL.distributed_training`:

## **Description**:
The `distributed_training` function is designed to handle distributed reinforcement learning (RL) training across multiple devices or workers. It supports various TensorFlow strategies, such as MirroredStrategy, MultiWorkerMirroredStrategy, and ParameterServerStrategy. The function is optimized for both single-node and multi-node setups, enabling distributed training with optional experience replay buffers, including prioritized and hindsight experience replay (HER). 

This function also supports parallel data collection through a **pool network** and optional just-in-time (JIT) compilation for performance optimization.

## **Parameters**:

- **`global_batch_size`** (`int`): The total batch size for training distributed across devices or workers.

- **`optimizer`** (`tf.keras.optimizers.Optimizer`): The optimizer used to update model parameters during training.

- **`strategy`** (`tf.distribute.Strategy`): A TensorFlow distribution strategy to manage the distributed training setup. This could be `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `ParameterServerStrategy`.

- **`episodes`** (`int`, optional): The number of training episodes to run. If set to `None`, the function will run indefinitely.

- **`num_episodes`** (`int`, optional): Alternative to `episodes`, used in specific strategy cases like `MultiWorkerMirroredStrategy`. Defaults to `None`.

- **`jit_compile`** (`bool`, optional, default=`True`): Whether to enable TensorFlow's Just-In-Time (JIT) compilation for performance optimization.

- **`pool_network`** (`bool`, optional, default=`True`): Whether to use a pool network for experiences collection.

- **`processes`** (`int`, optional): The number of parallel processes to use for data collection when `pool_network` is enabled. If set to `None`, multiprocessing is disabled.

- **`processes_her`** (`int`, optional): The number of parallel processes dedicated to Hindsight Experience Replay (HER) data collection, if HER is enabled.

- **`processes_pr`** (`int`, optional): The number of parallel processes for prioritized experience replay (PR) data collection, if PR is enabled.

- **`shuffle`** (`bool`, optional, default=`False`): If `True`, shuffles the data in the pool before training to prevent overfitting to recent experiences.

- **`p`** (`int`, optional): Controls how frequently to log intermediate results. If set to `None`, it defaults to `p=9`.

## **Returns**:
- **None**. The function logs training progress, including loss and reward information, at specified intervals. It may also save model parameters based on a given frequency.

## **Details**:

1. **Training with Distribution Strategies**:
   - The function adapts to various TensorFlow distribution strategies:
     - **`MirroredStrategy`**: For synchronous training across multiple GPUs on a single machine.
     - **`MultiWorkerMirroredStrategy`**: For synchronous training across multiple workers.
     - **`ParameterServerStrategy`**: For asynchronous training with parameter servers.

2. **Parallel Data Collection (Pool Network)**:
   - When `pool_network` is enabled, the function sets up parallel processes using Python's `multiprocessing` to collect experience (state, action, reward, next-state, done) from multiple environments. The data is stored in shared memory using multiprocessing managers.
   - The data can be used to update the agent’s neural network either through traditional replay or advanced methods like HER or prioritized replay.

3. **Handling HER and PR**:
   - If HER is enabled (`processes_her` is not `None`), the function initializes additional buffers and processes to handle HER-specific data collection.
   - Similarly, for prioritized replay (`processes_pr` is not `None`), the function maintains a TD-error (temporal difference error) list to prioritize experiences during replay.

4. **Training Execution**:
   - For each episode, the function collects experience using the pool network (if enabled) and updates the agent’s model parameters through the specified optimizer and distribution strategy. The loss is calculated either through a customized `train1` method (pool network) or `train2` method (direct training).
   - After every few episodes (controlled by `p`), the function logs the loss, reward, and progress. If a performance criterion is met (e.g., a certain average reward threshold), the training may terminate early.

5. **Model Saving**:
   - The function saves model parameters periodically, based on a pre-specified frequency (`save_freq`). If the parameter `save_param_only` is set, only model parameters are saved, otherwise the full model is saved.

6. **Time Tracking**:
   - The function keeps track of the total training time, logging it at the end of the training session.

## **Usage Example**:

```python
# Example usage of the distributed_training function
global_batch_size = 64
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
strategy = tf.distribute.MirroredStrategy()

agent.distributed_training(
    global_batch_size=global_batch_size, 
    optimizer=optimizer, 
    strategy=strategy, 
    episodes=100, 
    pool_network=True, 
    processes=8
)
```

In this example, the function runs distributed training using the `MirroredStrategy`, where experience is collected in parallel through 8 processes and stored in a pool buffer. Training runs for 100 episodes with a global batch size of 64.

# `Policy classes`:

## **SoftmaxPolicy**

### **Description**:
Implements a softmax policy for multinomial distribution. This policy selects actions based on a probability distribution, where each action has a probability proportional to the exponentiated Q-value or reward estimate.

### **Methods**:

- **`select_action(nb_actions, probs)`**:
  - **Arguments**:
    - `nb_actions` (`int`): Number of possible actions.
    - `probs` (`np.ndarray`): A probability distribution over the actions.
  - **Returns**:
    - `action` (`int`): The action selected according to the probability distribution `probs`.

### **Usage**:
Use `SoftmaxPolicy` with the `set` function of an RL agent:
```python
policy = SoftmaxPolicy()
model.set(policy=policy)
```

---

## **EpsGreedyQPolicy**

### **Description**:
Implements the epsilon-greedy policy. With probability `epsilon`, this policy selects a random action to encourage exploration. Otherwise, it selects the action with the highest estimated Q-value.

### **Constructor**:

- **`__init__(eps=.1)`**:
  - **Arguments**:
    - `eps` (`float`): Epsilon value representing the probability of choosing a random action. Defaults to `0.1`.

### **Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The selected action, either random (with probability `eps`) or the best action (with probability `1 - eps`).

### **Usage**:
Use `EpsGreedyQPolicy` with the `set` function of an RL agent:
```python
policy = EpsGreedyQPolicy(eps=0.05)
model.set(policy=policy)
```

---

## **GreedyQPolicy**

### **Description**:
Implements the greedy policy, where the agent always selects the action with the highest estimated Q-value. This policy does not explore other actions.

### **Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The action with the highest Q-value.

### **Usage**:
Use `GreedyQPolicy` with the `set` function of an RL agent:
```python
policy = GreedyQPolicy()
model.set(policy=policy)
```

---

## **BoltzmannQPolicy**

### **Description**:
Implements the Boltzmann Q Policy. This policy selects actions based on a probability distribution derived from exponentiated Q-values, where higher Q-values have higher probabilities. The `tau` parameter controls the exploration: higher `tau` values result in more exploration, while lower values focus on exploitation.

### **Constructor**:

- **`__init__(tau=1., clip=(-500., 500.))`**:
  - **Arguments**:
    - `tau` (`float`): Temperature parameter controlling exploration. Default is `1.0`.
    - `clip` (`tuple`): Range to clip the Q-values before exponentiation. Default is `(-500., 500.)`.

### **Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The selected action according to the Boltzmann distribution of Q-values.

### **Usage**:
Use `BoltzmannQPolicy` with the `set` function of an RL agent:
```python
policy = BoltzmannQPolicy(tau=0.5)
model.set(policy=policy)
```

---

## **MaxBoltzmannQPolicy**

### **Description**:
Combines epsilon-greedy and Boltzmann Q-policy. With probability `epsilon`, the agent follows the Boltzmann distribution to select an action. With probability `1 - epsilon`, it selects the action with the highest Q-value.

### **Constructor**:

- **`__init__(eps=.1, tau=1., clip=(-500., 500.))`**:
  - **Arguments**:
    - `eps` (`float`): Epsilon value for selecting random actions.
    - `tau` (`float`): Temperature parameter for Boltzmann exploration.
    - `clip` (`tuple`): Range to clip the Q-values. Default is `(-500., 500.)`.

### **Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The selected action, either based on Boltzmann exploration or the greedy choice.

### **Usage**:
Use `MaxBoltzmannQPolicy` with the `set` function of an RL agent:
```python
policy = MaxBoltzmannQPolicy(eps=0.1, tau=0.5)
model.set(policy=policy)
```

---

## **BoltzmannGumbelQPolicy**

### **Description**:
Implements the Boltzmann-Gumbel exploration policy, which is invariant to the mean of rewards but sensitive to reward variance. This policy uses Gumbel noise to perturb the Q-values for exploration and adapts over time based on the parameter `C`.

### **Constructor**:

- **`__init__(C=1.0)`**:
  - **Arguments**:
    - `C` (`float`): Exploration parameter to adjust for variance in rewards.

### **Methods**:

- **`select_action(q_values, step_counter)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
    - `step_counter` (`int`): Current step of the training process.
  - **Returns**:
    - `action` (`int`): The selected action based on Boltzmann-Gumbel exploration.

### **Usage**:
Use `BoltzmannGumbelQPolicy` with the `set` function of an RL agent:
```python
policy = BoltzmannGumbelQPolicy(C=1.0)
model.set(policy=policy)
```

---

## **GumbelSoftmaxPolicy**

### **Description**:
Implements the Gumbel Softmax policy for continuous action spaces. This policy samples from a Gumbel distribution and returns one-hot encoded actions for discrete action selection.

### **Constructor**:

- **`__init__(temperature=1.0, eps=0.01)`**:
  - **Arguments**:
    - `temperature` (`float`): Temperature parameter for Gumbel sampling.
    - `eps` (`float`): Epsilon value for exploration in the one-hot encoding process.

### **Methods**:

- **`onehot_from_logits(logits)`**:
  - **Arguments**:
    - `logits` (`np.ndarray`): The unnormalized log-probabilities (logits) for each action.
  - **Returns**:
    - `onehot_action` (`np.ndarray`): One-hot encoded action.

- **`sample_gumbel(shape, eps=1e-20)`**:
  - **Arguments**:
    - `shape` (`tuple`): Shape of the Gumbel sample to be drawn.
    - `eps` (`float`): Small epsilon to avoid numerical issues.
  - **Returns**:
    - `sample` (`np.ndarray`): Gumbel-distributed sample.

- **`gumbel_softmax_sample(logits)`**:
  - **Arguments**:
    - `logits` (`np.ndarray`): Logits for each action.
  - **Returns**:
    - `softmax_probs` (`np.ndarray`): Softmax probabilities for each action.

- **`gumbel_softmax(logits)`**:
  - **Arguments**:
    - `logits` (`np.ndarray`): Logits for each action.
  - **Returns**:
    - `y` (`np.ndarray`): One-hot encoded action sampled using Gumbel softmax.

### **Usage**:
Use `GumbelSoftmaxPolicy` with the `set` function of an RL agent:
```python
policy = GumbelSoftmaxPolicy(temperature=0.5, eps=0.01)
model.set(policy=policy)
```

---

This documentation provides detailed descriptions and usage examples for each policy class and reflects the typical way of passing these policies to an RL agent using the `set` function.

# `Noise classes`:

## **GaussianWhiteNoiseProcess**

### **Description**:
Implements a Gaussian white noise process, generating noise from a Gaussian distribution with mean `mu` and time-varying standard deviation `sigma` (which anneals over time). This type of noise is commonly used in exploration strategies for continuous action spaces.

### **Constructor**:
- **`__init__(mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1)`**:
  - **Arguments**:
    - `mu` (`float`): Mean of the Gaussian distribution.
    - `sigma` (`float`): Initial standard deviation.
    - `sigma_min` (`float`): Minimum standard deviation after annealing. If `None`, annealing is disabled.
    - `n_steps_annealing` (`int`): Number of steps over which the annealing occurs.
    - `size` (`int`): Size of the noise vector to be sampled.

### **Methods**:

- **`sample()`**:
  - Generates a sample of Gaussian noise based on the current standard deviation (`sigma`), which anneals over time.
  - **Returns**:
    - `sample` (`np.ndarray`): Sampled noise from the Gaussian distribution.

### **Usage**:
```python
noise = GaussianWhiteNoiseProcess(mu=0., sigma=1., sigma_min=0.1, n_steps_annealing=1000, size=1)
model.set(noise=noise)
```

---

## **OrnsteinUhlenbeckProcess**

### **Description**:
This process generates noise using the Ornstein-Uhlenbeck process, a continuous-time stochastic process often used to model time-correlated noise. It is frequently applied in reinforcement learning, especially for exploration in environments with continuous action spaces (e.g., DDPG). The noise tends to revert to the mean over time, controlled by the parameter `theta`.

### **Constructor**:
- **`__init__(theta, mu=0., sigma=1., dt=1e-2, size=1, sigma_min=None, n_steps_annealing=1000)`**:
  - **Arguments**:
    - `theta` (`float`): Rate of mean reversion (higher `theta` means stronger pull towards `mu`).
    - `mu` (`float`): Mean value to which the process reverts.
    - `sigma` (`float`): Initial standard deviation of the noise.
    - `dt` (`float`): Time step for discretization.
    - `size` (`int`): Size of the noise vector.
    - `sigma_min` (`float`): Minimum standard deviation for annealing.
    - `n_steps_annealing` (`int`): Number of steps over which the standard deviation anneals.

### **Methods**:

- **`sample()`**:
  - Generates a noise sample based on the current state of the process and updates the internal state.
  - **Returns**:
    - `x` (`np.ndarray`): Sampled noise from the Ornstein-Uhlenbeck process.

- **`reset_states()`**:
  - Resets the internal state (`x_prev`) of the process to a random value drawn from a Gaussian distribution.

### **Usage**:
```python
noise = OrnsteinUhlenbeckProcess(theta=0.15, mu=0., sigma=0.3, dt=1e-2, size=1)
model.set(noise=noise)
```

---

These noise processes, such as `GaussianWhiteNoiseProcess` and `OrnsteinUhlenbeckProcess`, are typically used to introduce randomness during action selection in continuous action space reinforcement learning algorithms like DDPG. You can set them up as the noise generator in your RL agent's `set` function.These processes help in efficient exploration by generating noise that is added to the agent’s actions during training.

---

# `Building a Custom Agent by Extending the RL Base Class`

This example demonstrates how to construct a reinforcement learning (RL) agent by extending a custom `RL` base class. The implementation uses both `Model` and `RL` classes to structure the agent modularly. Here, `Model` serves as a neural network wrapper, while `RL` manages RL-specific components.

## Step 1: Import `nn` and Define the Neural Network (Q-network) Class

In this step, we start by importing `nn` from `Note`, a module that provides layer utilities and parameter management. The `Qnet` class, which inherits from the `Model` base class, uses `nn` layers for efficient Q-network construction.

```python
from Note import nn

class Qnet(nn.Model):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
    
    def __call__(self, x):
        x = self.dense2(self.dense1(x))
        return x
```

Here, the `Model` superclass provides foundational methods for defining layers and managing parameters, making the setup of complex architectures more straightforward.

## Step 2: Create the DQN Agent by Extending the RL Class and Set Up the Environment

The `DQN` class represents the agent, inheriting core reinforcement learning functionalities by extending the `RL` base class.

The agent’s Q-network and target network, `q_net` and `target_q_net`, are constructed using the `Qnet` class, where `nn` provides functions to define dense layers, activations, and to handle parameters more conveniently. Additionally, `self.env` initializes the "CartPole-v0" environment from `gym` directly within the agent, so it has a predefined environment for interaction.

```python
import gym

class DQN(nn.RL):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.param = self.q_net.param  # Parameters managed by `nn`
        self.env = gym.make('CartPole-v0')  # Environment created within the agent class
    
    def action(self, s):
        return self.q_net(s)
    
    def __call__(self, s, a, next_s, r, d):
        a = tf.expand_dims(a, axis=1)
        q_value = tf.gather(self.q_net(s), a, axis=1, batch_dims=1)
        next_q_value = tf.reduce_max(self.target_q_net(next_s), axis=1)
        target = tf.cast(r, 'float32') + 0.98 * next_q_value * (1 - tf.cast(d, 'float32'))
        TD = (q_value - target)
        return tf.reduce_mean(TD ** 2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.param, self.param)
```

By using `nn`, we simplify neural network setup, making it easy to define and initialize layers and parameters across both networks. The `RL` base class enables managing RL-specific tasks like parameter updates and calculating the loss.

With the `RL` superclass, configuring and managing RL-specific methods, such as target network updates (`update_param`) and loss computation, becomes easier.

## Step 3: Initialize the Model and Train the Agent

After defining both `Qnet` and `DQN`, we can instantiate the agent, set hyperparameters, and begin training using the `RL` class’s `train` method. The `train` method simplifies the training loop and efficiently manages data collection and updates.

```python
import tensorflow as tf
from Note.RL import rl

model = DQN(4, 128, 2)
model.set(policy=rl.EpsGreedyQPolicy(0.01), pool_size=10000, batch=64, update_steps=10)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
``` 

This setup showcases how `nn`, `Model`, and `RL` components work together to streamline the development of reinforcement learning agents.
