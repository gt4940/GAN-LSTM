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
GAN_win_size = 1024
LSTM_win_size = 300
gen_size = 100000
noise_dim = 100
BUFFER_SIZE = 100000
BATCH_SIZE = 32

stress = preprocessing('./data/MechData.npz', 'ShearStress', GAN_win_size)
v = preprocessing('./data/MechData.npz', 'V', GAN_win_size)
train_data = np.concatenate((stress, v), axis=2)

dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = make_generator_model()
discriminator = make_discriminator_model()
gan_trainer = GANTrainer(generator, discriminator, noise_dim=noise_dim, batch_size=BATCH_SIZE, log_dir=log_dir)
gan_trainer.train(dataset, epochs=100)

image_generator = tf.saved_model.load(log_dir + '/generator')
temp_data = tf.random.normal([gen_size, noise_dim])
generated_image = image_generator(temp_data, training=False)

stress = preprocessing('./data/MechData.npz', 'ShearStress', LSTM_win_size)
v = preprocessing('./data/MechData.npz', 'V', LSTM_win_size)
ttf = preprocessing('./data/ttf_example.npy', 'TTF', LSTM_win_size)

dataset = split_data(stress, v, ttf, 'train').shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
valid_dataset = split_data(stress, v, ttf, 'valid').batch(BATCH_SIZE)
test_dataset = split_data(stress, v, ttf, 'test').batch(BATCH_SIZE)

lstm = make_LSTM_model()
lstm_trainer = LSTMTrainer(lstm, log_dir=log_dir)
lstm_trainer.train(dataset, valid_dataset, epochs=200)