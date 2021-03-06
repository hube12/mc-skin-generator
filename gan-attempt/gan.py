import numpy as np, matplotlib.pyplot as plt

# Change keras backend
import os ; os.environ["KERAS_BACKEND"] = "theano"#"plaidml.keras.backend" #"theano"

from keras import Sequential, Model, Input
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from keras.models import load_model

def main(epochs=500, batch_size=64):
    skin_dataset = np.load('../dataset/data.npy') # (5594, 16384)
    print(skin_dataset.shape)
    if False: # Load models
        generator = load_model('output/generator_final.h5')
        discriminator= load_model('output/discriminator_final.h5')
    else:
        generator = create_generator()
        discriminator = create_discriminator()
    
    gan = create_gan(discriminator, generator)
    
    for e in range(epochs):
        print(f'Epoch {e}')
        for _ in range(batch_size):
            noise = np.random.normal(0,1, [batch_size, 100])
            
            fake_skins = generator.predict(noise)
            real_skins = skin_dataset[np.random.randint(low=0, high=skin_dataset.shape[0], size=batch_size)]
            X = np.concatenate([real_skins, fake_skins])

            Y = np.zeros(2*batch_size)
            Y[:batch_size] = 0.9
            
            discriminator.trainable = True
            discriminator.train_on_batch(X, Y)
            discriminator.trainable = False

            X = np.random.normal(0,1, [batch_size, 100])
            Y = np.ones(batch_size)

            gan.train_on_batch(X, Y)

        if e % 5 == 0:
            plot_imgs(e, generator, discriminator)
            generator.save('output/generator_epoch.h5')
            discriminator.save('output/discriminator_epoch.h5')
    plot_imgs("final", generator, discriminator)
    generator.save('output/generator_final.h5')
    discriminator.save('output/discriminator_final.h5')


def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def create_generator():
    g = Sequential(name='generator')
    g.add(Dense(128, input_dim=100)) ; g.add(LeakyReLU(0.2))
    g.add(Dense(256))                ; g.add(LeakyReLU(0.2))
    g.add(Dense(512))               ; g.add(LeakyReLU(0.2))
    g.add(Dense(1024))               ; g.add(LeakyReLU(0.2))
    g.add(Dense(64*64*4,              activation='tanh'))
    g.compile(optimizer=adam_optimizer(), loss='binary_crossentropy')
    return g
def create_discriminator():
    d = Sequential(name='discriminator')
    d.add(Dense(1024, input_dim=64*64*4)) ; d.add(LeakyReLU(0.2)) ; d.add(Dropout(0.3))
    d.add(Dense(512))                    ; d.add(LeakyReLU(0.2)) ; d.add(Dropout(0.3))
    d.add(Dense(256))                     ; d.add(LeakyReLU(0.2)) ; d.add(Dropout(0.3))
    d.add(Dense(128))                     ; d.add(LeakyReLU(0.2))
    d.add(Dense(1,                         activation='sigmoid'))
    d.compile(optimizer=adam_optimizer(), loss='binary_crossentropy')
    return d
def create_gan(discriminator, generator):
    discriminator.trainable = False

    gan_input = Input((100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

def plot_imgs(epoch, generator, discriminator, examples=4, dim=(2,2), figsize=(5,5)):
    pics = []
    while len(pics) < examples:
        noise = np.random.normal(loc=0, scale=1, size=(1,100))
        generated_images = generator.predict(noise)
        output = discriminator.predict(generated_images)
        if output[0, 0] > .6:
            pics.append(generated_images)
    
    generated_images = np.array(pics)
    generated_images = generated_images.reshape(examples,64,64,4)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'output/gan_generated_image_{epoch}.png')

if __name__ == "__main__": main()