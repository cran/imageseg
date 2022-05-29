def down_conv_block(m, filter_mult, filters, kernel_size, name=None):
  m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
  m = layers.BatchNormalization()(m)
  
  m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
  m = layers.BatchNormalization(name=name)(m)
  
  return m
  
  # R
  down_conv_block <- function(m, filter_mult, filters, kernel_size, name=None){
  input %>% layer_conv_2d(filter_mult * filters, kernel_size, padding='same', activation='relu') %>%
  layer_batch_normalization() %>%
  
  layer_conv_2d(filter_mult * filters, kernel_size, padding='same', activation='relu') %>%
  layer_batch_normalization(name=name)
  }
  
  
  
  
  
  
  def up_conv_block(m, prev, filter_mult, filters, kernel_size, prev_2=None, prev_3=None, prev_4=None, name=None):
    m = layers.Conv2DTranspose(filter_mult * filters, kernel_size, strides=(2, 2), padding='same', activation='relu')(m)
  m = layers.BatchNormalization()(m)
  
  # Concatenate layers; varies between UNet and UNet++
  if prev_4 is not None:
    m = layers.Concatenate()([m, prev, prev_2, prev_3, prev_4])
  elif prev_3 is not None:
    m = layers.Concatenate()([m, prev, prev_2, prev_3])
  elif prev_2 is not None:
    m = layers.Concatenate()([m, prev, prev_2])
  else:
    m = layers.Concatenate()([m, prev])
  
  m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
  m = layers.BatchNormalization(name=name)(m)
  
  return m




  # R
  up_conv_block <- function(m, prev, filter_mult, filters, kernel_size, 
  prev_2=NULL, prev_3=NULL, prev_4=NULL, name=NULL){
    
    m_tmp <- m %>% 
    layer_conv_2d(filter_mult * filters, kernel_size, strides=(2, 2), padding='same', activation='relu') %>%
    layer_batch_normalization()
  
  # Concatenate layers; varies between UNet and UNet++
  if (!is.null(prev_4)) {
    m_tmp2 <- layer_concatenate([m_tmp, prev, prev_2, prev_3, prev_4])
    } else {
        if (!is.null(prev_3)) {
    m_tmp2 <- layer_concatenate([m, prev, prev_2, prev_3])
    } else {
      if (!is.null(prev_2)) {
    m_tmp2 <- layer_concatenate([m, prev, prev_2])
    } else {
    m_tmp2 <- layer_concatenate([m, prev])  
    }
    }
    }

  m_tmp2 %>% layer_conv_2d(filter_mult * filters, kernel_size, padding='same', activation='relu') %>%
  layer_batch_normalization(name=name)
  
  return m
}


  
  
  def build_unet(model_input, filters, kernel_size):
    # Downsampling / encoding portion
  conv0 = down_conv_block(model_input, 1, filters, kernel_size)
  pool0 = layers.MaxPooling2D((2, 2))(conv0)
  
  conv1 = down_conv_block(pool0, 2, filters, kernel_size)
  pool1 = layers.MaxPooling2D((2, 2))(conv1)
  
  conv2 = down_conv_block(pool1, 4, filters, kernel_size)
  pool2 = layers.MaxPooling2D((2, 2))(conv2)
  
  conv3 = down_conv_block(pool2, 8, filters, kernel_size)
  pool4 = layers.MaxPooling2D((2, 2))(conv3)
  
  # Middle of network
  conv4 = down_conv_block(pool4, 16, filters, kernel_size)
  
  # Upsampling / decoding portion
  uconv3 = up_conv_block(conv4, conv3, 8, filters, kernel_size)
  
  uconv2 = up_conv_block(uconv3, conv2, 4, filters, kernel_size)
  
  uconv1 = up_conv_block(uconv2, conv1, 2, filters, kernel_size)
  
  uconv0 = up_conv_block(uconv1, conv0, 1, filters, kernel_size)
  
  return uconv0
  
  
  def build_unet_plus_plus(model_input, filters, kernel_size, l):
    # Variables names follow the UNet++ paper: [successively downsampled layers_successively upsampled layers)
    # First stage of backbone: downsampling
    conv0_0 = down_conv_block(model_input, 1, filters, kernel_size, name='conv0_0')
  pool0_0 = layers.MaxPooling2D((2, 2))(conv0_0)
  conv1_0 = down_conv_block(pool0_0, 2, filters, kernel_size, name='conv1_0')
  
  if l > 1:
    # Second stage
    pool1_0 = layers.MaxPooling2D((2, 2))(conv1_0)
  conv2_0 = down_conv_block(pool1_0, 4, filters, kernel_size, name='conv2_0')
  
  if l > 2:
    # Third stage
    pool2_0 = layers.MaxPooling2D((2, 2))(conv2_0)
  conv3_0 = down_conv_block(pool2_0, 8, filters, kernel_size, name='conv3_0')
  
  if l > 3:
    # Fourth stage
    pool3_0 = layers.MaxPooling2D((2, 2))(conv3_0)
  conv4_0 = down_conv_block(pool3_0, 16, filters, kernel_size, name='conv4_0')
  
  # First stage of upsampling and skip connections
  conv0_1 = up_conv_block(conv1_0, conv0_0, 1, filters, kernel_size, name='conv0_1')
  out = conv0_1
  
  if l > 1:
    # Second stage
        conv1_1 = up_conv_block(conv2_0, conv1_0, 2, filters, kernel_size, name='conv1_1')
        conv0_2 = up_conv_block(conv1_1, conv0_1, 1, filters, kernel_size, prev_2=conv0_0, name='conv0_2')
        out = conv0_2

        if l > 2:
            # Third stage
            conv2_1 = up_conv_block(conv3_0, conv2_0, 4, filters, kernel_size, name='conv2_1')
            conv1_2 = up_conv_block(conv2_1, conv1_1, 2, filters, kernel_size, prev_2=conv1_0, name='conv1_2')

            conv0_3 = up_conv_block(conv1_2, conv0_2, 1, filters, kernel_size, prev_2=conv0_1, prev_3=conv0_0,
                                    name='conv0_3')
            out = conv0_3

            if l > 3:
                # Fourth stage
                conv3_1 = up_conv_block(conv4_0, conv3_0, 8, filters, kernel_size, name='conv3_1')
                conv2_2 = up_conv_block(conv3_1, conv2_1, 4, filters, kernel_size, prev_2=conv2_0, name='conv2_2')
                conv1_3 = up_conv_block(conv2_2, conv1_2, 2, filters, kernel_size, prev_2=conv1_1, prev_3=conv1_0,
                                        name='conv1_3')
                conv0_4 = up_conv_block(conv1_3, conv0_3, 1, filters, kernel_size, prev_2=conv0_2, prev_3=conv0_1,
                                        prev_4=conv0_0, name='conv0_4')
                out = conv0_4

    return out
