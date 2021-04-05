import keras,os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from keras.preprocessing import image
from keras.datasets import fashion_mnist,cifar10,cifar100,mnist
from keras.utils import to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = " 2"


def generator(input_shape):
    inputs = Input(input_shape)
    x = Dense(128 * 16 * 16)(inputs)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(3, 7, activation='tanh', padding='same')(x)
    return Model(inputs, x)


def discriminator(input_shape):
    inputs = Input(input_shape)

    x = Conv2D(128, 3)(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)

    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)  # 分类层

    return Model(inputs, x)


dis = discriminator((32,32,3))
dis.compile(loss=keras.losses.binary_crossentropy,optimizer= keras.optimizers.RMSprop(lr = 0.0008,clipvalue = 1.0,decay=1e-8))


(x_train,y_train),(x_test,y_test)= cifar10.load_data()

y_train_label = y_train
y_test_label = y_test

x_train = x_train[y_train.flatten() == 7]  #选择马类数据即可
x_train = x_train.reshape(5000,32,32,3).astype('float32')/255.

epochs = 4000
batch_size = 64
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

generated_img = []
discriminator_loss = []
generator_loss = []
save_dir = './A-GAN-PHOTO'

for epoch in range(epochs):

    noise = np.random.normal(0, 1, size=(batch_size, 100))
    img_index = np.random.randint(0, 5000, batch_size)
    fake_img = gen.predict(noise)
    real_img = x_train[img_index]
    data = np.concatenate([fake_img, real_img])
    label = np.concatenate([fake, valid])
    label += 0.05 * np.random.random(label.shape)

    d_loss = dis.train_on_batch(data, label)

    # ---------------------
    #  训练生成模型
    # ---------------------

    noise_ = np.random.normal(0, 1, size=(batch_size, 100))
    g_loss = gan.train_on_batch(noise_, valid)

    if epoch % 100 == 0:
        im = fake_img[0]
        generated_img.append(im)
        img = image.array_to_img(fake_img[0] * 255, scale=False)
        img.save(os.path.join(save_dir, 'generated_horse' + str(epoch) + '.png'))  # 保存一张生成图像

        img = image.array_to_img(real_img[0] * 255, scale=False)
        img.save(os.path.join(save_dir, 'real_horse' + str(epoch) + '.png'))  # 保存一张真实图像用于对比

        print('discriminator_loss:', d_loss)
        print('adversal_loss:', g_loss)
        discriminator_loss.append(d_loss)
        generator_loss.append(g_loss)

        # discriminator_loss.append(d_loss[-1])
        # generator_loss.append(g_loss[-1])
        # print("d_loss:%f"%d_loss[-1])
        # print("g_loss:%f"%g_loss[-1])
        print("epoch:%d" % epoch + "========")



fig, axes = plt.subplots(nrows=2, ncols=20, sharex=True, sharey=True, figsize=(80,12))
imgs = generated_img

for image, row in zip([imgs[:20], imgs[20:40]], axes):
    for img, ax in zip(image, row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)

plt.plot(discriminator_loss,label='discriminator_loss')
plt.plot(generator_loss,label='generator_loss')
plt.legend()
