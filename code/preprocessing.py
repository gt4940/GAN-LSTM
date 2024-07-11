import numpy as np
import tensorflow as tf
from scipy import io

def mat2npz(path):
    mat_file = io.loadmat(path)

    mat_file.pop('__header__', None)
    mat_file.pop('__version__', None)
    mat_file.pop('__globals__', None)

    np.savez('MechData.npz', **mat_file)
    return

def cal_TTF():
    return

def make_dataset():
    return

def normalization(array): # -1<=X<=1
    max_ = np.max(array)
    min_ = np.min(array)
    norm_1 = (max_ + min_)/2
    norm_2 = (max_ - min_)/2
    norm_array = (array - norm_1) / norm_2
    return norm_array


def make_window(array, win_size):
    new_window = np.zeros((array.shape[0] - win_size, win_size))
    for i in range(array.shape[0] - win_size):
        new_window[i] = array[i:i + win_size]
    new_window = new_window[:, :, np.newaxis]
    return new_window


def preprocessing(file_path, feature, win_size):
    if feature == 'TTF':
        array = np.load(file_path)
    else:
        with np.load(file_path) as npz:
            array = npz[feature][200000:300000 + win_size]
    array = array.reshape(-1)
    # Normalization
    norm_array = normalization(array)
    # Make window
    data_window = make_window(norm_array, win_size)
    return data_window


def split_data(x_1, x_2, y_1, dataset_type):
    if dataset_type == 'train':
        random = np.random.permutation(80000)
        x_1 = x_1[random]
        x_2 = x_2[random]
        y_1 = y_1[random]
    elif dataset_type == 'valid':
        x_1 = x_1[80000:90000]
        x_2 = x_2[80000:90000]
        y_1 = y_1[80000:90000]
    elif dataset_type == 'test':
        x_1 = x_1[90000:]
        x_2 = x_2[90000:]
        y_1 = y_1[90000:]
    else:
        print("dataset_type error")
    
    x = np.concatenate((x_1, x_2), axis=2)
    dataset = make_dataset(x, y_1)
    return dataset


def make_dataset(x, y):
    features = tf.convert_to_tensor(x)
    labels = tf.convert_to_tensor(y)
    features_dataset = tf.data.Dataset.from_tensor_slices(features)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    return dataset

##################################################################


"""def make_train_dataset(dds, gen):
    S_gen, T_gen = data_gen()
    S_dds, T_dds = data_dds('train')

    S_gen, T_gen = S_gen[:gen], T_gen[:gen]
    S_dds, T_dds = S_dds[:dds], T_dds[:dds]

    X_train = np.concatenate((S_gen, S_dds), axis=0)
    Y_train = np.concatenate((T_gen, T_dds), axis=0)
    dataset = make_dataset(X_train, Y_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset

def make_vaild_dataset():
    S_dds, T_dds = data_dds('valid')
    dataset = make_dataset(S_dds, T_dds).batch(BATCH_SIZE)
    return dataset

def make_test_data():
    S_dds, T_dds = data_dds('test')
    return S_dds, T_dds"""