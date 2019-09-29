import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from data import get_celebA, flags
from model import get_generator, get_discriminator
from PIL import Image
num_tiles = int(np.sqrt(flags.sample_size))

  
def train():
    images, images_path = get_celebA(flags.output_size, flags.n_epoch, flags.batch_size)
    G = get_generator([None, flags.z_dim])
    D = get_discriminator([None, flags.output_size, flags.output_size, flags.c_dim])

    G.train()
    D.train()

    d_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)
    g_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)

    n_step_epoch = int(len(images_path) // flags.batch_size)
    
    # Z = tf.distributions.Normal(0., 1.)
    
    for epoch in range(flags.n_epoch):
        for step, batch_images in enumerate(images):
            if batch_images.shape[0] != flags.batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                # z = Z.sample([flags.batch_size, flags.z_dim]) 
                z = np.random.normal(loc=0.0, scale=1.0, size=[flags.batch_size, flags.z_dim]).astype(np.float32)
                d_logits = D(G(z))
                d2_logits = D(batch_images)
                # discriminator: real images are labelled as 1
                drealw = np.random.normal(loc=0.7, scale=1.2, size=[flags.batch_size, flags.z_dim]).astype(np.float32)
                d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
                # discriminator: images from generator (fake) are labelled as 0
                dfakew = np.random.normal(loc=0.0, scale=0.3, size=[flags.batch_size, flags.z_dim]).astype(np.float32)
                d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
                # combined loss for updating discriminator
                d_loss = d_loss_real + d_loss_fake
                # generator: try to fool discriminator to output 1
                g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            del tape

            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(epoch, \
                  flags.n_epoch, step, n_step_epoch, time.time()-step_time, d_loss, g_loss))
        
        if np.mod(epoch, flags.save_every_epoch) == 0:
            G.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')
            D.save_weights('{}/D.npz'.format(flags.checkpoint_dir), format='npz')
            G.eval()
            result = G(z)
            G.train()
            #tl.visualize.save_images(result.numpy(), [num_tiles, num_tiles], '{}/train_{:02d}.png'.format(flags.sample_dir, epoch))
            arr=result.numpy()
            arr=arr[:num_tiles*num_tiles]
            chops=np.reshape(arr,(num_tiles,num_tiles,arr.shape[1],arr.shape[2],arr.shape[3]))
            H = np.cumsum([x[0].shape[0] for x in chops])
            W = np.cumsum([x.shape[1] for x in chops[0]])
            DD = chops[0][0]
            recon = np.empty((H[-1], W[-1], DD.shape[2]), DD.dtype)
            for rd, rs in zip(np.split(recon, H[:-1], 0), chops):
                for d, s in zip(np.split(rd, W[:-1], 1), rs):
                    d[...] = s
            rescaled = (255.0 / recon.max() * (recon - recon.min())).astype(np.uint8)
            im = Image.fromarray(rescaled)
            im.save('{}/train_{:02d}.png'.format(flags.sample_dir, epoch))
if __name__ == '__main__':
    train()
