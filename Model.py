import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.models import Model


latent_dim = 200
input_shape = (448, 448, 3)

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def initialize_encoder(input_shape: tuple) -> Model:
    """
    Initialize the Encoder Neural Network
    """
    #input shape (448, 448, 3)
    input_image = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation="relu")(input_image)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same', activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same', activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])

    # Encoder Build
    encoder = Model(input_image, [z_mean, z_log_var, z], name="encoder")

    print("✅ encoder initialized")

    return encoder

def initialize_decoder() -> Model:

    latent_inputs = Input(shape=(latent_dim,))
    y = Dense(7*7*128, activation='tanh')(latent_inputs)
    y = Reshape((7, 7, 128))(y)

    y = Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation="relu")(y)
    y = Conv2DTranspose(256, (3, 3), strides=1, padding='same', activation="relu")(y)
    y = Conv2DTranspose(256, (3, 3), strides=1, padding='same', activation="relu")(y)
    y = Conv2DTranspose(256, (3, 3), strides=1, padding='same', activation="relu")(y)

    y = Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation="relu")(y)
    y = Conv2DTranspose(128, (3, 3), strides=1, padding='same', activation="relu")(y)
    y = Conv2DTranspose(128, (3, 3), strides=1, padding='same', activation="relu")(y)
    y = Conv2DTranspose(128, (3, 3), strides=1, padding='same', activation="relu")(y)

    y = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation="relu")(y)
    y = Conv2DTranspose(64, (3, 3), strides=1, padding='same', activation="relu")(y)
    y = Conv2DTranspose(64, (3, 3), strides=1, padding='same', activation="relu")(y)

    y = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation="relu")(y)
    y = Conv2DTranspose(32, (3, 3), strides=1, padding='same', activation="relu")(y)
    y = Conv2DTranspose(32, (3, 3), strides=1, padding='same', activation="relu")(y)

    y = Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation="relu")(y)
    y = Conv2DTranspose(16, (3, 3), strides=1, padding='same', activation="relu")(y)
    y = Conv2DTranspose(16, (3, 3), strides=1, padding='same', activation="relu")(y)

    decoder_output = Conv2DTranspose(3, (3, 3), strides=2, padding='same', activation='sigmoid')(y)

    # Decoder Build
    decoder = Model(inputs=latent_inputs, outputs=decoder_output, name="decoder")

    print("✅ decoder initialized")

    return decoder


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# ## Instantiate Model and Load weights

    # @staticmethod
    # def create_vae(encoder, decoder, **kwargs):
    #     encoder = initialize_encoder(input_shape=input_shape)
    #     decoder = initialize_decoder()
    #     vae = VAE(encoder=encoder, decoder=decoder, **kwargs)
    #     return vae
