import matplotlib.pyplot as plt
import gym
import numpy as np
from PIL import Image  # To transform the image in the Processor

# Convolutional Backbone Network
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam

# Keras-RL
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

env = gym.make("air_raid:air_raid-v0")
# env.reset()
# for s in range(20):
#     env.render("human")
#     action = env.action_space.sample()
#     img, reward, done, info = env.step(action)
#     plt.figure()
#     plt.imshow(img)
#     plt.pause(10)
#
#
nb_actions = env.action_space.n

WINDOW_LENGTH = 10
size_x = env.frame_size_x
size_y = env.frame_size_y
IMG_SHAPE = [size_x, size_y]


class ImageProcessor(Processor):
    def process_observation(self, observation):
        # First convert the numpy array to a PIL Image
        img = Image.fromarray(observation)
        # Then resize the image
        img = img.resize(IMG_SHAPE)
        # And convert it to grayscale  (The L stands for luminance)
        # Convert the image back to a numpy array and finally return the image
        img = np.array(img)
        return img.astype("uint8")  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We divide the observations by 255 to compress it into the intervall [0, 1].
        # This supports the training of the network
        # We perform this operation here to save memory.
        processed_batch = batch.astype("float32") / 255.0
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1, 1)


input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1], 3)

# ###### creating the model
model = Sequential()
model.add(Permute((4, 2, 3, 1), input_shape=input_shape))

model.add(Convolution2D(32, (8, 8), strides=(2, 2), kernel_initializer="he_normal"))
model.add(Activation("relu"))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation("relu"))
model.add(Convolution2D(64, (4, 4), strides=(1, 1)))
model.add(Activation("relu"))
model.add(Convolution2D(32, (8, 8), strides=(1, 1)))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dense(nb_actions))
model.add(Activation("linear"))
print(model.summary())


# model.load_weights("test_dqn_air_raid_weights_20000.h5f")
memory = SequentialMemory(limit=40000, window_length=WINDOW_LENGTH)

processor = ImageProcessor()

policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr="eps",
    value_max=1.0,
    value_min=0.1,
    value_test=0.25,
    nb_steps=40000,
)

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    policy=policy,
    memory=memory,
    processor=processor,
    nb_steps_warmup=5000,
    gamma=0.99,
    target_model_update=5000,
    train_interval=WINDOW_LENGTH,
    delta_clip=1,
)

dqn.compile(Adam(learning_rate=0.000001), metrics=["mae"])

checkpoint_weights_filename = "test_dqn_" + "air_raid" + "_weights_{step}.h5f"
checkpoint_callback = ModelIntervalCheckpoint(
    checkpoint_weights_filename, interval=5000
)

history = dqn.fit(
    env,
    nb_steps=50000,
    callbacks=[checkpoint_callback],
    log_interval=10000,
    visualize=True,
)
x = history.epoch
y = (history.history)["episode_reward"]
plt.figure()
plt.plot(x, y)
plt.show()


weights_filename = "test_dqn_air_raid.h5f"
dqn.save_weights(weights_filename, overwrite=True)


dqn.test(env, nb_episodes=8, visualize=True)
