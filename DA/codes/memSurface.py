from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import matplotlib.pyplot as plt
import math

import tensorflow as tf
Nimages = 8

os.environ["CUDA_DEVICES_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  

  

  input_image = tf.cast(image, tf.float32)
  

  return input_image

def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  

  return input_image

def normalize(input_image):
  input_image = (input_image / 127.5) - 1
  

  return input_image

def random_crop(input_image):
  stacked_image = tf.stack([input_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[1, IMG_HEIGHT, IMG_WIDTH, 1])

  return cropped_image[0]

#@tf.function()
def random_jitter(input_image):
  # resizing to 286 x 286 x 3
  input_image = resize(input_image, 256, 256)

  # randomly cropping to 256 x 256 x 3
  #input_image = random_crop(input_image)

  #if tf.random.uniform(()) > 0.5:
    # random mirroring
  #  input_image = tf.image.flip_left_right(input_image)
    

  return input_image

def load_image_train(image_file):
  #for i in range(Nimages):
  #  image_file.find('.')
  #print("file_path: ",bytes.decode(image_file),type(bytes.decode(image_file)))
  #name = PATH+image_file+image_ext#tf.strings.format("{}{}.jpg", (PATH,image_file))
  #print(name)
  
  input_image = load(image_file)
  input_image = random_jitter(input_image)
  #input_image = normalize(input_image)

  return input_image

def load_image_test(image_file):
  input_image = load(image_file)
  input_image = resize(input_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image = normalize(input_image)

  return input_image

#PATH='/home/loki/Lakshmi/anomaly_detection/images/'
PATH='/media/313727ac-f148-4b50-9224-3f3986e68217/Lakshmi/anomaly_detection/DA/train/images/'
BUFFER_SIZE = 100/Nimages
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def loadImgSeq(i):
  img_set = tf.Variable(tf.zeros([1,Nimages,IMG_HEIGHT,IMG_HEIGHT,1]))
  for j in range(Nimages):
    #img = PATH+'frame'+str(i*Nimages+j+1)+'.jpg'
    img = PATH+str(i*Nimages+j+1)+'.jpg'
    image = load_image_train(img)
    
    img_set[0,j].assign(image)
    
  return img_set


def downsample(filters, size, strides, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 100)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv3D(filters, size, strides, activation='sigmoid',padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  #result.add(tf.keras.layers.sigmoid())

  return result

def upsample(filters, size, strides, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 100)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv3DTranspose(filters, size, strides,
                                    padding='same',activation='sigmoid',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  #result.add(tf.keras.layers.BatchNormalization())

  #if apply_dropout:
  #    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

OUTPUT_CHANNELS = 1

def Generator():
  

  down_stack = [
    downsample(64, [Nimages,1,1], [1,1,1], apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(1, [Nimages,1,1], [Nimages,1,1]), # (bs, 64, 64, 128)
    #downsample(1, 1, [2,1,1]), # (bs, 64, 64, 128)
    
  ]
  
  up_stack = [
    upsample(64, [Nimages,1,1], [Nimages,1,1]), # (bs, 128, 128, 128)
    #upsample(64, 1, [2,1,1]), # (bs, 128, 128, 128)
  ]
  
  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS, 1,
                                         strides=[1,1,1],
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)
  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[Nimages,256,256,1])
  x = inputs
  
  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    print(x.shape)
    skips.append(x)
  
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    #x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

generator.summary()

def generate_images(model, test_input, tar):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  prediction = model(test_input, training=True)
  
  x = generator.get_layer("sequential_1")
  intermediate_layer_model = tf.keras.Model(inputs = generator.input, outputs = x.output)
  intermediate_output = intermediate_layer_model(test_input)
  
  
  
  plt.figure(figsize=(15,15))
  
  #print(intermediate_output.shape)
  display_list = [prediction[0,0,:,:,0], prediction[0,1,:,:,0], intermediate_output[0,0,:,:,0]]
  title = ['Predicted Image', 'Predicted Image', 'Predicted Image', 'Predicted Image', 'Intermediate Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.gray()
    #plt.savefig('books_read.png')
    plt.axis('off')
  plt.show()
  
  return prediction[0,0,:,:,0], prediction[0,1,:,:,0],intermediate_output[0,0,:,:,0]

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


import numpy as np
def generator_loss(gen_output, target):
  
  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  #l1_loss = tf.reduce_mean(tf.norm(target-gen_output))
  lbda = 1
  total_gen_loss = (l1_loss)
  
  return total_gen_loss


#@tf.function
def train_step(reg_loss,input_image, target):
  
  with tf.GradientTape() as gen_tape:
    gen_output = generator(input_image, training=True)
    
    #disc_real_output = discriminator([input_image, target], training=True)
    #disc_generated_output = discriminator([input_image, gen_output], training=True)
    lbda = 1
    gen_loss = reg_loss+lbda*generator_loss(gen_output, input_image)
    #disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_loss,
                                          generator.trainable_variables)
  #discriminator_gradients = disc_tape.gradient(disc_loss,
  #                                             discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  #discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
  #                                           discriminator.trainable_variables))


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_mem_DA_sparse_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator)

def train(epochs):
  
  for epoch in range(epochs):
    start = time.time()
    
    for iter in range(0,100):
      input_image = loadImgSeq(iter)
      #lum_img = input_image[0,0,:,:,0]
      #plt.imshow(lum_img/255.0)
      #plt.show()
      
      rho = 0.01
      target = loadImgSeq(iter)
      x = generator.get_layer("sequential_1")
      intermediate_layer_model = tf.keras.Model(inputs = generator.input, outputs = x.output)
      intermediate_output = intermediate_layer_model(target)

      reg_loss = 0
      img = np.array((intermediate_output[0,0,:,:,0]))
             
      for i in range(IMG_WIDTH):
        for j in range(IMG_HEIGHT):
          img[i][j] = np.abs(img[i][j])
          if (rho/(0.1+img[i][j])) != 0:            
             term1 = rho*np.log(rho/(0.1+img[i][j]))
            # print(rho/(0.1+img[i][j])) 
          else:
             term1 = 0
          if (1-rho)/(0.1+(1-img[i][j])) != 0:
             term2 = (1-rho)*np.log((1-rho)/(0.1+(1-img[i][j])))
          else:
             term2 = 0
          reg_loss = term1+term2
      train_step(0.01*reg_loss,input_image, input_image)
    #for input_image in dataset:
    #  train_step(input_image)

    #clear_output(wait=True)
    #for inp, tar in test_dataset.take(1):s
    input_image = loadImgSeq(10)
    #generate_images(generator, input_image, input_image)

    # saving (checkpoint) the model every 20 epochs
    if (epoch) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))


train(100)
