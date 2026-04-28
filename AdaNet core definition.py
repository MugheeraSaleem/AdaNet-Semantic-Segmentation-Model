# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:43:40 2026

@author: mughe
"""

import tensorflow as tf
import math
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Lambda, Concatenate, Input 
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from collections import defaultdict
import numpy as np
import os
import time
from tensorflow.keras.callbacks import Callback
import albumentations as A
from tensorflow.python.profiler import option_builder
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa
from matplotlib.patches import Patch
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Workaround for OpenMP conflict
OptionalDim = Union[int, tf.Tensor, None]
Number = Union[float, int]
Stride = Union[Number, Tuple[Number, Number]]
CHANNELS_FIRST = 'channels_first'

# @tf.keras.utils.register_keras_serializable()
def compute_adaptive_span_mask(threshold: tf.float32,
                               ramp_softness: tf.float32,
                               pos: tf.Tensor) -> tf.Tensor:
  """Adaptive mask as proposed in https://arxiv.org/pdf/1905.07799.pdf.

  Args:
    threshold: Threshold that starts the ramp.
    ramp_softness: Smoothness of the ramp.
    pos: Position indices.

  Returns:
   A tf.Tensor<tf.complex64> containing the
   thresholdings for the mask with the same size of pos.
  """
  output = (1.0 / ramp_softness) * (ramp_softness + threshold - pos)
  return tf.cast(tf.clip_by_value(output, 0.0, 1.0), dtype=tf.complex64)


# @tf.keras.utils.register_keras_serializable()
class StrideConstraint(tf.keras.constraints.Constraint):
    """Constraint strides.
      
    Strides are constrained in [1,+infty) as default as smoothness factor
    always leave some feature map by default.
"""
    def __init__(self,
               lower_limit: Optional[float] = None,
               upper_limit: Optional[float] = None,
               **kwargs):
        """Constraint strides.
        
        Args:
          lower_limit: Lower limit for the stride.
          upper_limit: Upper limit for the stride.
          **kwargs: Additional arguments for parent class.
        """
        super().__init__(**kwargs)
        self._lower_limit = lower_limit if lower_limit is not None else 1.0
        self._upper_limit = (
            upper_limit if upper_limit is not None else tf.float32.max)

    def __call__(self, kernel):
          return tf.clip_by_value(kernel, self._lower_limit, self._upper_limit)
    def get_config(self):
        config = super().get_config()
        config.update({
            'lower_limit': self.lower_limit,
            'upper_limit': self.upper_limit
        })
        return config


# @tf.keras.utils.register_keras_serializable()
class StrideRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, lambda_reg=0.1):
        """Regularizer for stride parameters.
        
        Args:
            lambda_reg: Regularization strength factor (default: 0.1)
        """
        self.lambda_reg = lambda_reg

    def __call__(self, stride_tensor):
        """Compute regularization loss from stride tensor.
        
        Args:
            stride_tensor: Tensor containing stride values (shape: [2] for [S_h, S_w])
        """
        # Compute product of stride dimensions (S_h * S_w)
        stride_product = tf.reduce_prod(stride_tensor)
        
        # Calculate inverse stride regularization term
        inv_stride = 1.0 / stride_product
        return self.lambda_reg * inv_stride

    def get_config(self):
        """Get configuration for serialization."""
        return {'lambda_reg': self.lambda_reg}
    

# @tf.keras.utils.register_keras_serializable()
class DiffStride(tf.keras.layers.Layer):
    """Learnable Spectral pooling layer, computed in the Fourier domain.
    
    The adaptive window function is inspired from
    https://arxiv.org/pdf/1905.07799.pdf.
    """
    
    def __init__(self,
                 strides: Stride = (2.0, 2.0),
                 lambda_reg=0.01,
                 smoothness_factor: float = 4.0,
                 cropping: bool = True,
                 trainable: bool = True,
                 shared_stride: bool = False,
                 lower_limit_stride: Optional[float] = None,
                 upper_limit_stride: Optional[float] = None,
                 data_format: str = CHANNELS_FIRST,
                 **kwargs,):
      """Learnable Spectral pooling layer.
    
      Vertical and horizontal positions are the indices of the feature map. It
      allows to selectively weight the output of the fourier transform based
      on these positions.
      Args:
        strides: Fractional strides to init before learning the reduction in the
          Fourier domain.
        smoothness_factor: Smoothness factor to reduce/crop the input feature map
          in the Fourier domain.
        cropping: Boolean to specify if the layer crops or set to 0 the
          coefficients outside the cropping window in the Fourier domain.
        trainable: Boolean to specify if the stride is learnable.
        shared_stride: If `True`, a single parameter is shared for vertical and
          horizontal strides.
        lower_limit_stride: Lower limit for the stride. It can be useful when
          there are memory issues, it avoids the stride converge to small values.
        upper_limit_stride: Upper limit for the stride.
        data_format: either `channels_first` or `channels_last`. Be aware that
          channels_last will increase the memory cost due transformation to
          channels_first.
        **kwargs: Additional arguments for parent class.
      """
      super().__init__(**kwargs)
       
      self.stride_regularizer = StrideRegularizer(lambda_reg)
      self.lambda_reg = lambda_reg  # Store lambda_reg as instance variable
      self._cropping = cropping
      self._smoothness_factor = smoothness_factor
      self._shared_stride = shared_stride
      self.trainable = trainable
      self._lower_limit_stride = lower_limit_stride
      self._upper_limit_stride = upper_limit_stride
      self.data_format = data_format  # Store original data_format string
      self._channels_first = (data_format == CHANNELS_FIRST)
    
      # Ensures a tuple of floats.
      strides = (
          (strides, strides) if isinstance(strides, (int, float)) else strides)
      strides = tuple(map(float, strides))
      if strides[0] != strides[1] and shared_stride:
        raise ValueError('shared_stride requires the same initialization for '
                         f'vertical and horizontal strides but got {strides}')
      if strides[0] < 1 or strides[1] < 1:
        raise ValueError(f'Both strides should be >=1 but got {strides}')
      if smoothness_factor < 0.0:
        raise ValueError('Smoothness factor should be >= 0 but got '
                         f'{smoothness_factor}.')
      self._strides = strides
    
    def build(self, input_shape):
      del input_shape
      init = self._strides[0] if self._shared_stride else self._strides
      self.strides = self.add_weight(
          shape=(1,) if self._shared_stride else (2,),
          initializer=tf.initializers.Constant(init),
          trainable=self.trainable,
          dtype=tf.float32,
          name='strides',
          constraint=StrideConstraint(
              lower_limit=self._lower_limit_stride,
              upper_limit=self._upper_limit_stride),
          regularizer=self.stride_regularizer
          )
      
    def call(self, inputs: tf.Tensor, training: bool = False):
        if not self._channels_first:
            inputs = tf.transpose(inputs, (0, 3, 1, 2))
        
        # ========== Start of Added Code ==========
        # Track input dimensions as tensors
        self.input_H = tf.shape(inputs)[2]
        self.input_W = tf.shape(inputs)[3]
        self.input_C = tf.shape(inputs)[1]
        
        # Calculate output dimensions using CURRENT strides
        self.output_H = tf.cast(self.input_H, tf.float32) / self.strides[0]
        self.output_W = tf.cast(self.input_W, tf.float32) / self.strides[1]
        
        # Convert to integers when eager execution is enabled
        if tf.executing_eagerly():
            self.input_H = int(self.input_H.numpy())
            self.input_W = int(self.input_W.numpy())
            self.input_C = int(self.input_C.numpy())
            self.output_H = int(self.output_H.numpy())
            self.output_W = int(self.output_W.numpy())
        # ========== End of Added Code ==========
    
        batch_size, channels = inputs.shape.as_list()[:2]
        height, width = self.input_H, self.input_W  # Use tracked dimensions
        
        horizontal_positions = tf.range(width // 2 + 1, dtype=tf.float32)
        vertical_positions = tf.range(
            height // 2 + height % 2, dtype=tf.float32)
        vertical_positions = tf.concat([
            tf.reverse(vertical_positions[(height % 2):], axis=[0]),
            vertical_positions], axis=0)
        
        # Stride constraints logic
        min_vertical_stride = tf.cast(height, tf.float32) / (
            tf.cast(height, tf.float32) - self._smoothness_factor)
        min_horizontal_stride = tf.cast(width, tf.float32) / (
            tf.cast(width, tf.float32) - self._smoothness_factor)
        
        if self._shared_stride:
            min_stride = tf.math.maximum(min_vertical_stride, min_horizontal_stride)
            self.strides[0].assign(tf.math.maximum(self.strides[0], min_stride))
            vertical_stride, horizontal_stride = self.strides[0], self.strides[0]
        else:
            self.strides[0].assign(
                tf.math.maximum(self.strides[0], min_vertical_stride))
            self.strides[1].assign(
                tf.math.maximum(self.strides[1], min_horizontal_stride))
            vertical_stride, horizontal_stride = self.strides[0], self.strides[1]
        
        # Apply constraints
        vertical_stride = self.strides.constraint(vertical_stride)
        horizontal_stride = self.strides.constraint(horizontal_stride)
        
        # Fourier operations
        f_inputs = tf.signal.rfft2d(inputs)
        horizontal_mask = compute_adaptive_span_mask(
            self.output_W/2 + 1, self._smoothness_factor, horizontal_positions)
        vertical_mask = compute_adaptive_span_mask(
            self.output_H/2, self._smoothness_factor, vertical_positions)
        
        vertical_mask = tf.signal.fftshift(vertical_mask)
        output = f_inputs * horizontal_mask[None, None, None, :]
        output = output * vertical_mask[None, None, :, None]
        
        if self._cropping:
            horizontal_to_keep = tf.stop_gradient(
                tf.where(tf.cast(horizontal_mask, tf.float32) > 0.)[:, 0])
            vertical_to_keep = tf.stop_gradient(
                tf.where(tf.cast(vertical_mask, tf.float32) > 0.)[:, 0])
            
            output = tf.gather(output, indices=vertical_to_keep, axis=2)
            output = tf.gather(output, indices=horizontal_to_keep, axis=3)
        
        result = tf.ensure_shape(
            tf.signal.irfft2d(output), [batch_size, channels, None, None])
        
        if not self._channels_first:
            result = tf.transpose(result, (0, 2, 3, 1))
            
        return result
          
    def compute_output_shape(self, input_shape):
        batch_size, channels = input_shape[:2]
        return (batch_size, channels, None, None)
    def get_config(self):
        # Serialize layer initialization parameters
        config = super().get_config().copy()
        config.update({                
            'strides': self._strides,  # Use initial stride values
            'lambda_reg': self.lambda_reg,
            'smoothness_factor': self._smoothness_factor,
            'cropping': self._cropping,
            'trainable': self.trainable,
            'shared_stride': self._shared_stride,
            'data_format': self.data_format  # Use stored data_format
        })
        return config


# @tf.keras.utils.register_keras_serializable()
class CUFHyperNetwork(tf.keras.layers.Layer):
    '''
    The hypernetwork learns to predict upsampling kernels for an input scale. 
    This has been implemented from the Continuous Upsampling Filters paper 
    @ https://arxiv.org/pdf/2210.06965 
    '''
    def __init__(self, output_channels, num_units=32, dct_basis=25, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels  # This is the number of input channels (C_in)
        self.num_units = num_units
        self.dct_basis = dct_basis
        self.kernel_size = 3  # For a 3x3 filter; hence, 9 values per location

    def build(self, input_shape):
        # A 4-layer MLP as described in the paper
        self.dense_layers = [
            tf.keras.layers.Dense(self.num_units, activation='relu',kernel_initializer='he_normal'),
            tf.keras.layers.Dense(self.num_units, activation='relu',kernel_initializer='he_normal'),
            # tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(self.num_units, activation='relu',kernel_initializer='he_normal'),
            tf.keras.layers.Dense(self.num_units, activation='relu',kernel_initializer='he_normal')
        ]
        # Final layer outputs 9 * output_channels values (one weight per kernel position per channel)
        self.kernel_output = tf.keras.layers.Dense(
            self.output_channels * (self.kernel_size ** 2),
            activation='linear'
        )
        super().build(input_shape)

    def call(self, delta, scale, kernel_indices):
        # Encode delta: [H, W, 2] -> [H, W, 2 * dct_basis]
        delta_enc = self._dct_encode(delta, self.dct_basis, f_max=2.0)
        
        # Encode scale: [2] -> [1, 2 * dct_basis] and broadcast to [H, W, 2 * dct_basis]
        scale = tf.reshape(scale, [1, 2])
        scale_enc = self._dct_encode(scale, self.dct_basis, f_max=2.0)
        scale_enc = tf.broadcast_to(scale_enc, tf.shape(delta_enc))
        
        # Encode kernel indices: [H, W, 9, 2] -> [H, W, 9, 2 * 9] then average over the 9 positions
        kernel_enc = self._dct_encode(kernel_indices, dct_basis=9, f_max=1.0)
        kernel_enc = tf.reduce_mean(kernel_enc, axis=2)  # Shape: [H, W, 18]
        
        # Concatenate along the channel axis: result shape [H, W, (2*dct_basis + 2*dct_basis + 18)]
        x = tf.concat([delta_enc, scale_enc, kernel_enc], axis=-1)
        
        # Process through the MLP
        for layer in self.dense_layers:
            x = layer(x)
        
        # Final dense layer outputs a tensor of shape [H, W, output_channels * 9]
        out = self.kernel_output(x)
        # Reshape to get separate kernel weights per filter element:
        out = tf.reshape(
            out,
            [tf.shape(delta)[0], tf.shape(delta)[1], self.kernel_size ** 2, self.output_channels]
        )
        return out

    def _dct_encode(self, x, dct_basis, f_max):
        frequencies = tf.linspace(1.0, f_max, dct_basis)
        frequencies = tf.sort(frequencies)
        x_exp = tf.expand_dims(x, axis=-1)  # Now shape becomes [..., D, 1]
        arguments = 2 * math.pi * (2 * x_exp + 1) * frequencies / 2
        # Reshape so that the last dimension becomes (dct_basis * D)
        return tf.reshape(
            tf.cos(arguments),
            tf.concat([tf.shape(x)[:-1], [dct_basis * x.shape[-1]]], axis=0)
        )
                                                   
# @tf.keras.utils.register_keras_serializable()
class CUFLayer(tf.keras.layers.Layer):
    '''
    The CUF model adopted as a feature map upsampling layer for upsampling 
    the featuremaps AdaNet model receives from the downsampling path or the previous decoder module.
    '''
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.hyper_network = None
        self.proj_conv = None

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError("CUFLayer requires 2 inputs: [main_input, skip_connection]")
        C_in = input_shape[0][-1]
        self.hyper_network = CUFHyperNetwork(output_channels=C_in, dct_basis=25)
        self.proj_conv = tf.keras.layers.Conv2D(self.filters, 1, padding='same')
        super().build(input_shape)

    def call(self, inputs):
        main_input, skip_connection = inputs
        batch_size = tf.shape(main_input)[0]
        H_in = tf.shape(main_input)[1]
        W_in = tf.shape(main_input)[2]
        H_target = tf.shape(skip_connection)[1]
        W_target = tf.shape(skip_connection)[2]

        # Calculate scale
        scale_h = tf.cast(H_target, tf.float32) / tf.cast(H_in, tf.float32)
        scale_w = tf.cast(W_target, tf.float32) / tf.cast(W_in, tf.float32)
        scale = tf.stack([scale_h, scale_w], axis=-1)  # Shape [2]

        # Generate delta grid [H_target, W_target, 2]
        grid_h = tf.linspace(0.0, 1.0, H_target)
        grid_w = tf.linspace(0.0, 1.0, W_target)
        grid_h, grid_w = tf.meshgrid(grid_h, grid_w, indexing='ij')
        delta = tf.stack([grid_h, grid_w], axis=-1)

        # Generate kernel indices [H_target, W_target, 9, 2]
        # De-normalized Kernel_indices
        # kernel_indices = tf.constant(
        #     [[[i, j] for j in range(self.kernel_size)] 
        #      for i in range(self.kernel_size)], 
        #     dtype=tf.float32
        # )  # [3, 3, 2]
        
        
        # Normalized Kernel_indices
        kernel_indices = tf.constant(
            [[[i - 1, j - 1]  # Normalized to [-1, 1]
              for j in range(self.kernel_size)]
             for i in range(self.kernel_size)], 
            dtype=tf.float32
        )
        
        kernel_indices = tf.reshape(kernel_indices, [1, 1, -1, 2])  # [1, 1, 9, 2]
        kernel_indices = tf.tile(kernel_indices, [H_target, W_target, 1, 1])  # [H_target, W_target, 9, 2]

        # Predict kernels via hypernetwork [H_target, W_target, 9, C_in]
        kernels = self.hyper_network(delta, scale, kernel_indices)

        # Depth-wise convolution with predicted kernels
        upsampled = self._cuf_upsample(main_input, kernels, H_target, W_target)
        return self.proj_conv(upsampled)

    def _cuf_upsample(self, x, kernels, H_target, W_target):
        # Step 1: Upsample input features to target resolution using nearest-neighbor
        x_upsampled = tf.image.resize(x, [H_target, W_target], method='nearest')
        
        # Step 2: Extract patches from upsampled features
        patches = tf.image.extract_patches(
            images=x_upsampled,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        
        # Reshape patches: [B, H_target, W_target, kernel_size^2, C_in]
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(
            patches,
            [batch_size, H_target, W_target, self.kernel_size**2, -1]
        )
        
        # Step 3: Apply per-pixel dynamic kernels (generated by hyper-network)
        # Kernels shape: [H_target, W_target, kernel_size^2, C_in]
        # Expand for batch: [1, H_target, W_target, kernel_size^2, C_in]
        kernels_expanded = tf.expand_dims(kernels, axis=0)        
        # Step 4: Depth-wise convolution via element-wise multiplication + sum
        output = tf.reduce_sum(patches * kernels_expanded, axis=3)
        return output
        

# =============================================================================
# AdaNet that uses separate CUF layers for upsampling.
# =============================================================================
class AdaNet(tf.keras.Model):
    '''
    3 layer AdaNet model, with 3 encoder modules for dynamic downsampling in encoder 
    and 3 decoder modules for smart/scale-aware upsampling in the decoder. The two are 
    connected by a simple bottleneck. 
    '''
    def __init__(self, input_shape=(256, 256, 3), num_classes=1):
        super(AdaNet, self).__init__()
        # ---------------------------------------------------------------------
        # Downsampling Path 
        # ---------------------------------------------------------------------
        # Block 1
        self.conv1_1 = tf.keras.layers.Conv2D(64, 3, padding='same', 
                                              activation='relu', data_format='channels_last')
        self.conv1_2 = tf.keras.layers.Conv2D(64, 3, padding='same', 
                                              activation='relu', data_format='channels_last')
        self.coarse_drop1 = tf.keras.layers.SpatialDropout2D(0.1)
        self.diffstride1 = DiffStride(
            strides=(2.0, 2.0),
            smoothness_factor=2.0,
            lambda_reg=0.0075, 
            cropping=True,
            trainable=True,
            shared_stride=False,
            data_format='channels_last'
        )
        
        # Block 2
        self.conv2_1 = tf.keras.layers.Conv2D(128, 3, padding='same', 
                                              activation='relu', data_format='channels_last')
        self.conv2_2 = tf.keras.layers.Conv2D(128, 3, padding='same', 
                                              activation='relu', data_format='channels_last')
        self.coarse_drop2 = tf.keras.layers.SpatialDropout2D(0.1)
        self.diffstride2 = DiffStride(
            strides=(2.0, 2.0),
            smoothness_factor=2.0,
            lambda_reg=0.0075,
            cropping=True,
            trainable=True,
            shared_stride=False,
            data_format='channels_last'
        )
        
        # Block 3
        self.conv3_1 = tf.keras.layers.Conv2D(256, 3, padding='same', 
                                              activation='relu', data_format='channels_last')
        self.conv3_2 = tf.keras.layers.Conv2D(256, 3, padding='same', 
                                              activation='relu', data_format='channels_last')
        self.coarse_drop3 = tf.keras.layers.SpatialDropout2D(0.1)
        self.diffstride3 = DiffStride(
            strides=(2.0, 2.0),
            smoothness_factor=2.0,
            lambda_reg=0.0075,
            cropping=True,
            trainable=True,
            shared_stride=False,
            data_format='channels_last'
        )
        # Bottleneck
        self.conv_b1 = tf.keras.layers.Conv2D(512, 3, padding='same', 
                                              activation='relu', data_format='channels_last')
        self.conv_b2 = tf.keras.layers.Conv2D(512, 3, padding='same', 
                                              activation='relu', data_format='channels_last')
        self.bottleneck_drop = tf.keras.layers.SpatialDropout2D(0.3)
        
        # ---------------------------------------------------------------------
        # Upsampling Path: separate CUF layers per block.
        # ---------------------------------------------------------------------
        # Block 1: Upsample -> fuse with s3
        self.cuf_upsample1 = CUFLayer(filters=256)
        self.concat1 = tf.keras.layers.Concatenate()
        self.conv_u1_1 = tf.keras.layers.Conv2D(256, 3, padding='same', 
                                                activation='relu', data_format='channels_last')
        self.conv_u1_2 = tf.keras.layers.Conv2D(256, 3, padding='same', 
                                                activation='relu', data_format='channels_last')
        self.up_drop1 = tf.keras.layers.SpatialDropout2D(0.1)
        
        # Block 2: Upsample -> fuse with s2
        self.cuf_upsample2 = CUFLayer(filters=128)
        self.concat2 = tf.keras.layers.Concatenate()
        self.conv_u2_1 = tf.keras.layers.Conv2D(128, 3, padding='same', 
                                                activation='relu', data_format='channels_last')
        self.conv_u2_2 = tf.keras.layers.Conv2D(128, 3, padding='same', 
                                                activation='relu', data_format='channels_last')
        self.up_drop2 = tf.keras.layers.SpatialDropout2D(0.1)
        
        # Block 3: Upsample -> fuse with s1
        self.cuf_upsample3 = CUFLayer(filters=64)
        self.concat3 = tf.keras.layers.Concatenate()
        self.conv_u3_1 = tf.keras.layers.Conv2D(64, 3, padding='same', 
                                                activation='relu', data_format='channels_last')
        self.conv_u3_2 = tf.keras.layers.Conv2D(64, 3, padding='same', 
                                                activation='relu', data_format='channels_last')
        self.up_drop3 = tf.keras.layers.SpatialDropout2D(0.1)
        
        # Output layer
        self.output_conv = tf.keras.layers.Conv2D(num_classes, 1, 
                                                  activation='softmax', data_format='channels_last')
    
    def call(self, inputs, training=False):
        # ---------------
        # Downsampling Path
        # ---------------
        # Block 1
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.coarse_drop1(x, training=training)
        s1 = x  # skip connection 1
        x = self.diffstride1(x, training=training)
        
        # Block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.coarse_drop2(x, training=training)
        s2 = x  # skip connection 2
        x = self.diffstride2(x, training=training)
        
        # Block 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.coarse_drop3(x, training=training)
        s3 = x  # skip connection 3
        x = self.diffstride3(x, training=training)
        
        # Bottleneck
        x = self.conv_b1(x)
        x = self.conv_b2(x)
        x = self.bottleneck_drop(x, training=training)
        
        # ---------------
        # Upsampling Path
        # ---------------
        # Block 1: Upsample from bottleneck to resolution of s3
        x = self.cuf_upsample1([x, s3])
        x = self.concat1([x, s3])
        x = self.conv_u1_1(x)
        x = self.conv_u1_2(x)
        x = self.up_drop1(x, training=training)
        
        # Block 2: Upsample to resolution of s2
        x = self.cuf_upsample2([x, s2])
        x = self.concat2([x, s2])
        x = self.conv_u2_1(x)
        x = self.conv_u2_2(x)
        x = self.up_drop2(x, training=training)
        
        # Block 3: Upsample to resolution of s1
        x = self.cuf_upsample3([x, s1])
        x = self.concat3([x, s1])
        x = self.conv_u3_1(x)
        x = self.conv_u3_2(x)
        x = self.up_drop3(x, training=training)
        
        # Output segmentation map
        return self.output_conv(x)