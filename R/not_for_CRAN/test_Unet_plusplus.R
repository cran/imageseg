down_conv_block <- function(input, filter_mult, filters, kernel_size, name=NULL){
  input %>%   layer_conv_2d(filters = filter_mult * filters, 
                kernel_size = kernel_size, 
                padding='same', 
                activation='relu') %>%
    layer_batch_normalization() %>%
    
    layer_conv_2d(filters = filter_mult * filters, 
                  kernel_size = kernel_size, 
                  padding='same', 
                  activation='relu') %>%
    layer_batch_normalization(name = name)
}



up_conv_block <- function(input, prev, filter_mult, filters, kernel_size, 
                          prev_2=NULL, prev_3=NULL, prev_4=NULL, name=NULL){
  
  m_tmp <- input %>% 
    layer_conv_2d(filter_mult * filters, kernel_size, strides= c(2, 2), padding='same', activation='relu') %>%
    layer_batch_normalization()
  
  # Concatenate layers; varies between UNet and UNet++
  if (!is.null(prev_4)) {
    m_tmp2 <- layer_concatenate(m_tmp, prev, prev_2, prev_3, prev_4)
  } else {
    if (!is.null(prev_3)) {
      m_tmp2 <- layer_concatenate(m_tmp, prev, prev_2, prev_3)
    } else {
      if (!is.null(prev_2)) {
        m_tmp2 <- layer_concatenate(m_tmp, prev, prev_2)
      } else {
        m_tmp2 <- layer_concatenate(list(m_tmp, prev))
      }
    }
  }
  
  m_tmp2 %>% layer_conv_2d(filter_mult * filters, kernel_size, padding='same', activation='relu') %>%
    layer_batch_normalization(name=name)
  
  return (m_tmp2)
}

net_h <- 128
net_w <- 128
channels <- 3
input_shape <- c(net_h, net_w, channels)

input_img <- layer_input(shape = input_shape,
                         name = 'input_img')



conv0_0 <- input_img %>% 
  down_conv_block(filter_mult = 1, filters = 16, kernel_size = 3, name = "conv0_0")

up_conv_block(input = tmp, prev = conv0_0, filter_mult = 1, filters = 16, kernel_size = 3)
