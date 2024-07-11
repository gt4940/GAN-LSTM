import os
import time
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

class GANTrainer:
    def __init__(self, generator, discriminator, noise_dim, batch_size, log_dir):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.batch_size = batch_size

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(5e-5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

        self.log_dir = log_dir
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.seed = tf.random.normal([self.batch_size, self.noise_dim])

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss, disc_loss

    def generate_and_save_images(self, model, epoch, test_input):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(20, 15))

        for i in range(predictions.shape[0]):
            plt.subplot(8, 8, 2*i+1)
            plt.plot(predictions[i, :, 0], 'C0')
            plt.subplot(8, 8, 2*i+2)
            plt.plot(predictions[i, :, 1], 'C1')

        plt.savefig(self.log_dir + f'/image_at_epoch_{epoch:04d}.png')
        plt.close()
        # plt.show()


    def train(self, dataset, epochs):
        now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for epoch in range(epochs):
            start = time.time()
            for image_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch)
            
            with self.summary_writer.as_default():
                tf.summary.scalar('gen_loss', gen_loss, step=epoch)
                tf.summary.scalar('disc_loss', disc_loss, step=epoch)
                tf.summary.scalar('total_loss', gen_loss + disc_loss, step=epoch)


            self.generate_and_save_images(self.generator, epoch + 1, self.seed)
            if (epoch + 1) % 5 == 0:
                self.checkpoint.save(file_prefix=self.log_dir + '/check_point')
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # 마지막 에포크가 끝난 후 생성합니다.
        self.generate_and_save_images(self.generator, epochs, self.seed)

        tf.saved_model.save(self.generator, self.log_dir + '/generator')
        tf.saved_model.save(self.discriminator, self.log_dir + '/discriminator')

class LSTMTrainer:
    def __init__(self, model, log_dir):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model = model
        self.log_dir = log_dir

    def train(self, train_dataset, valid_dataset, epochs):
        self.model.compile(optimizer=self.optimizer, loss='mse')
        history = self.model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)
        self.model.save(self.log_dir + '/LSTM_model_save.h5')