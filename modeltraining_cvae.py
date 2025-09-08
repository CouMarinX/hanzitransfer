import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import ast

# 数据文件路径
image_files = ["output/new_character1.txt", "output/synth_target.txt"]
label_files = ["output/radical_labels1.txt", "output/synth_radical_labels.txt"]

# 加载二维数组数据
def load_array(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    data = np.array(ast.literal_eval(content), dtype=np.float32)
    return data

# 合并数据
images = np.concatenate([load_array(f) for f in image_files], axis=0)
labels = np.concatenate([load_array(f) for f in label_files], axis=0)

# 预处理
x_train = images.reshape((-1, 64, 64, 1))
labels = labels.astype("int32")
num_radicals = int(labels.max()) + 1
cond_train = tf.keras.utils.to_categorical(labels, num_classes=num_radicals)

latent_dim = 16

# 编码器
encoder_inputs = layers.Input(shape=(64, 64, 1))
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
mu = layers.Dense(latent_dim)(x)
log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    mu, log_var = args
    epsilon = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * epsilon

z = layers.Lambda(sampling)([mu, log_var])

cond_input = layers.Input(shape=(num_radicals,))
z_cond = layers.Concatenate()([z, cond_input])

# 解码器
decoder_inputs = layers.Input(shape=(latent_dim + num_radicals,))
y = layers.Dense(32 * 32 * 32, activation="relu")(decoder_inputs)
y = layers.Reshape((32, 32, 32))(y)
y = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(y)
decoder_outputs = layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")(y)

decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

outputs = decoder(z_cond)

# CVAE 模型
cvae = Model([encoder_inputs, cond_input], outputs, name="cvae")

# KL 散度
kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
cvae.add_loss(kl_loss)

cvae.compile(optimizer="adam", loss="binary_crossentropy")

# 训练
cvae.fit([x_train, cond_train], x_train, epochs=10, batch_size=32, verbose=1)

# 保存解码器
decoder.save("hanzi_cvae_decoder.keras")
