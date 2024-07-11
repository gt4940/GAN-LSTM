import os
import datetime
import tensorflow as tf
import numpy as np
import random
from code.preprocessing import preprocessing, split_data
from code.train import GANTrainer, LSTMTrainer
from code.model import make_discriminator_model, make_generator_model, make_LSTM_model

# preset
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

log_dir = os.path.join('./logs', datetime.datetime.now().strftime("%m%d-%H%M%S"))
win_size = 1024
path = './data/MechData.npz'
BUFFER_SIZE = 100000
BATCH_SIZE = 32

stress = preprocessing(path, 'ShearStress', win_size)
v = preprocessing(path, 'V', win_size)
train_data = np.concatenate((stress, v), axis=2)

dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = make_generator_model()
discriminator = make_discriminator_model()
gan_trainer = GANTrainer(generator, discriminator, noise_dim=100, batch_size=32, log_dir=log_dir)
gan_trainer.train(dataset, epochs=10)

ttf = preprocessing('./data/ttf_example.npy', 'TTF', win_size)

BATCH_SIZE = 32
dataset = split_data(stress, v, ttf, 'train').shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
valid_dataset = split_data(stress, v, ttf, 'valid').batch(BATCH_SIZE)

lstm = make_LSTM_model()
lstm_trainer = LSTMTrainer(lstm, log_dir='./logs')
lstm_trainer.train(dataset, valid_dataset, epochs=200)