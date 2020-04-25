from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import tensorflow as tf
Nimages = 8

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
      tf.keras.layers.Conv3D(filters, size, strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, strides, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 100)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv3DTranspose(filters, size, strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

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





generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './backup'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator)

def loadImgSeq_test(i):
  
  PATH='/media/313727ac-f148-4b50-9224-3f3986e68217/Lakshmi/anomaly_detection/DA/test/fight/memIn/'
  img_set = tf.Variable(tf.zeros([1,Nimages,IMG_HEIGHT,IMG_HEIGHT,1]))
  for j in range(Nimages):
    #img = PATH+'frame'+str(i*Nimages+j+1)+'.jpg'
    img = PATH+str(i+j)+'.jpg'
    image = load_image_train(img)
    
    img_set[0,j].assign(image)
    
  return img_set

import numpy as np
import cv2 
j = 0
#checkpoint_dir = '/content/drive/My Drive/training_checkpoints'
#checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                 generator=generator)
#status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
checkpoint_dir = '/media/313727ac-f148-4b50-9224-3f3986e68217/Lakshmi/anomaly_detection/DA/model/training_mem_DA_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator)
                                 
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
PATH='/media/313727ac-f148-4b50-9224-3f3986e68217/Lakshmi/anomaly_detection/DA/test/fight/memOut/'
for iter in range(0,1000):
  input_image = loadImgSeq_test(iter)
  img1, img2, img3 = generate_images(generator, input_image, input_image)
  name = PATH+str(j)+'.png'
  j = j+1
  
  #plt.imshow(img3 * 0.5 + 0.5)
  #plt.gray()
  
  #plt.axis('off')
  
  print("plotting {}".format(iter))


  dpi = 80
  
  

    # What size does the figure need to be in inches to fit the image?
  figsize = IMG_WIDTH / float(dpi), IMG_HEIGHT / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
  fig = plt.figure(figsize=figsize)
  
  fig = plt.figure(figsize=figsize)
  ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
  ax.axis('off')

    # Display the image.
  ax.imshow(img3, cmap='gray')
  fig.savefig(name)
  #display_list = [img1, img2, img3]
  #title = ['Predicted Image', 'Predicted Image', 'Predicted Image', 'Predicted Image', 'Intermediate Image']

  #for i in range(3):
  #  plt.subplot(1, 3, i+1)
  #  plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
  #  plt.imshow(display_list[i] * 0.5 + 0.5)
  #  plt.gray()
    #plt.savefig(name)
  #  plt.axis('off')
  #plt.show()
  
  
  
  #plt.show()
