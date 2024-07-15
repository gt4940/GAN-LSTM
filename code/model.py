import tensorflow as tf

def make_generator_model():
    inputs = tf.keras.Input(shape=(100,))
    x = inputs
    x = tf.keras.layers.Dense(64*256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Reshape((64, 256))(x)
    x = tf.keras.layers.Conv1DTranspose(256, 5, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Conv1DTranspose(128, 5, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Conv1DTranspose(64, 5, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Conv1DTranspose(2, 5, strides=2, padding='same', use_bias=False, activation='tanh')(x)
    outputs = x
    model = tf.keras.Model(inputs, outputs)
    # model.summary()
    return model


def make_discriminator_model():
    inputs = tf.keras.Input(shape=(1024, 2))
    x = inputs
    x = tf.keras.layers.Conv1D(32, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv1D(64, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv1D(128, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv1D(256, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    outputs = x
    model = tf.keras.Model(inputs, outputs)
    # model.summary()
    return model


def make_LSTM_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=(300, 2), return_sequences=True))
    model.add(tf.keras.layers.LSTM(256))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    return model