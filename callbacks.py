# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:24:42 2026

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

# This file contatins all the training callbacks written to monitor strides, resulting feature map sizes and FLOPs during training 


#########################################################
#########################################################
#########################################################


class StrideMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} Stride Values:")
        print("Block 1 - Vertical:", self.model.diffstride1.strides[0].numpy(),
              "Horizontal:", self.model.diffstride1.strides[1].numpy())
        print("Block 2 - Vertical:", self.model.diffstride2.strides[0].numpy(),
              "Horizontal:", self.model.diffstride2.strides[1].numpy())
        print("Block 3 - Vertical:", self.model.diffstride3.strides[0].numpy(),
              "Horizontal:", self.model.diffstride3.strides[1].numpy())

class StrideHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.stride1_vert = []
        self.stride1_horz = []
        self.stride2_vert = []
        self.stride2_horz = []
        self.stride3_vert = []
        self.stride3_horz = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Record stride values
        self.stride1_vert.append(self.model.diffstride1.strides[0].numpy())
        self.stride1_horz.append(self.model.diffstride1.strides[1].numpy())
        self.stride2_vert.append(self.model.diffstride2.strides[0].numpy())
        self.stride2_horz.append(self.model.diffstride2.strides[1].numpy())
        self.stride3_vert.append(self.model.diffstride3.strides[0].numpy())
        self.stride3_horz.append(self.model.diffstride3.strides[1].numpy())
        
        # Add to logs for potential access (optional)
        logs['stride1_vert'] = self.stride1_vert[-1]
        logs['stride1_horz'] = self.stride1_horz[-1]
        logs['stride2_vert'] = self.stride2_vert[-1]
        logs['stride2_horz'] = self.stride2_horz[-1]
        logs['stride3_vert'] = self.stride3_vert[-1]
        logs['stride3_horz'] = self.stride3_horz[-1]


class DiffStrideShapeTracker(tf.keras.callbacks.Callback):
    def __init__(self, input_shape, log_interval=5):
        super().__init__()
        self.input_shape = input_shape
        self.layer_shapes = defaultdict(list)
        self.log_interval = log_interval
        self.history = []  # Stores (epoch, [area1, area2, area3])
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_interval != 0:
            return
            
        dummy_input = tf.zeros((1,) + self.input_shape)
        current_tensor = dummy_input
        areas = []
        
        for layer in self.model.layers:
            if isinstance(layer, DiffStride):
                input_shape = current_tensor.shape
                output_tensor = layer(current_tensor, training=False)
                output_shape = output_tensor.shape
                
                # Store shapes
                self.layer_shapes[layer.name].append((input_shape, output_shape))
                current_tensor = output_tensor
                
                # Print immediately
                print(f"\nEpoch {epoch+1} - {layer.name}:")
                print(f"Input shape: {tuple(input_shape[1:])} (HWC)")
                print(f"Output shape: {tuple(output_shape[1:])} (HWC)")
                # Corrected indices based on data format
                if layer.data_format == 'channels_last':
                    h = output_tensor.shape[1]
                    w = output_tensor.shape[2]
                else:
                    h = output_tensor.shape[2]
                    w = output_tensor.shape[3]
                areas.append(float(h * w))
                current_tensor = output_tensor
                
        self.history.append((epoch + 1, areas))


def plot_downsampling_comparison(shape_history, fixed_sizes=[128, 64, 32], epoch_stride=5,original_size=256):
    """
    Plots a grouped bar chart comparing fixed maxpool outputs with dynamic diffstride outputs.
    
    For each selected epoch (e.g., every 5 epochs), the function creates a group with six bars:
      - The first three (left side) show the fixed maxpool outputs for Block 1, Block 2, and Block 3.
      - The next three (right side) show the dynamic diffstride outputs (computed as sqrt(area)).
    
    Each block gets a distinct color. The text labels above the bars are offset enough to avoid merging.
    
    Parameters:
      shape_history: List of tuples (epoch, [area1, area2, area3]) where each "area" is the area (height×width)
                     of the block's output.
      fixed_sizes: The fixed outputs for maxpool (assumed square), e.g., [128, 64, 32].
      epoch_stride: Plot only every `epoch_stride` epochs.
    """
    # Filter history for epochs that match the epoch stride.
    filtered_history = [entry for entry in shape_history if entry[0] % epoch_stride == 0]
    if not filtered_history:
        print("No history entries available for the specified epoch stride.")
        return

    epochs = [entry[0] for entry in filtered_history]
    
    # For each diffstride block, compute effective output dimension (assume square output = sqrt(area))
    dynamic_dims = []
    for epoch, areas in filtered_history:
        dims = [np.sqrt(area) for area in areas]
        dynamic_dims.append(dims)
    
    n_epochs = len(epochs)
    n_blocks = 3  # three downsampling blocks per epoch
    n_bars_per_group = n_blocks * 2  # first half fixed, second half dynamic
    bar_width = 0.8
    group_gap = 1.5  # gap between epoch groups

    # Calculate x positions for each bar in every group.
    x_positions = []  # list (length = n_epochs) of lists (length = n_bars_per_group)
    for i in range(n_epochs):
        group_start = i * (n_bars_per_group + group_gap)
        # Positions: first three for fixed, next three for dynamic.
        xs = [group_start + j for j in range(n_bars_per_group)]
        x_positions.append(xs)
    
    # Colors for each block (you can adjust these as desired)
    block_colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    fig, ax = plt.subplots(figsize=(max(10, n_epochs * 2), 6))
    
    # Plot fixed maxpool bars (first three in each group)
    for i, xs in enumerate(x_positions):
        for block in range(n_blocks):
            x_val = xs[block]
            height = fixed_sizes[block]
            ax.bar(x_val, height, width=bar_width, color=block_colors[block], edgecolor='black', alpha=0.8)
            # Place text above the fixed bar with vertical offset (here, 2 pixels above the bar)
            ax.text(x_val, height + 2, f"{fixed_sizes[block]}x{fixed_sizes[block]}", 
                    ha='center', va='bottom', fontsize=9)
    
    # Plot dynamic diffstride bars (next three in each group)
    for i, xs in enumerate(x_positions):
        for block in range(n_blocks):
            x_val = xs[block + n_blocks]  # dynamic bars occupy the next positions
            height = dynamic_dims[i][block]
            ax.bar(x_val, height, width=bar_width, color=block_colors[block], edgecolor='black', 
                   alpha=0.6, hatch='//')
            ax.text(x_val, height + 2, f"{int(round(height))}x{int(round(height))}", 
                    ha='center', va='bottom', fontsize=9)
    
    # Set x-ticks: one tick per group, centered in the group.
    group_centers = [np.mean(xs) for xs in x_positions]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"Epoch {ep}" for ep in epochs])
    ax.set_ylabel("Feature Map Dimension (pixels)")
    ax.set_title("Fixed MaxPool vs Dynamic DiffStride Downsampling Comparison")
    ax.set_ylim(0, original_size)
    
    # Create legends.
    # Legends for block colors.
    block_legend = [Patch(facecolor=block_colors[i], label=f"Block {i+1}") for i in range(n_blocks)]
    # Legends for the bar styles.
    fixed_patch = Patch(facecolor='gray', label='Fixed (MaxPool)', alpha=0.8)
    dynamic_patch = Patch(facecolor='gray', label='Dynamic (DiffStride)', alpha=0.6, hatch='//')
    leg1 = ax.legend(block_legend, [f"Block {i+1}" for i in range(n_blocks)], loc='upper left')
    leg2 = ax.legend([fixed_patch, dynamic_patch], ['Fixed (MaxPool)', 'Dynamic (DiffStride)'], loc='upper right')
    ax.add_artist(leg1)
    
    plt.tight_layout()
    plt.savefig('D:/Mugheera/Full_CUF_4070_256x256/300_size_comparison_tversky_model_FULL_CUF_more_dropout_fully_non_reg_0.0075_strided_4070.png')
    plt.show()
    
    
input_shp = (256, 256, 3)

def get_flops(model, batch_size=1):
    tf.config.run_functions_eagerly(True)
    dummy_input = tf.ones((batch_size, input_shp[0], input_shp[1], input_shp[2]))
    _ = model(dummy_input)
    
    current_shape = input_shp
    total_flops = 0
    flops_breakdown = {
        'DiffStride': 0,
        'Conv2D': 0,
        'BatchNorm': 0,
        'Concatenate': 0,
        'OutputConv': 0,
        'Softmax': 0,
        'Activations': 0,
        'BiasAdd': 0,
        'CUF': 0
    }
    
    for layer in model.layers:
        if isinstance(layer, DiffStride):
            H_out = max(1, layer.output_H)
            W_out = max(1, layer.output_W)
            C = layer.input_C  
            
            fft_flops = 5 * C * H_out * W_out * (math.log(H_out, 2) + math.log(W_out, 2))
            masking_flops = 6 * C * H_out * ((W_out // 2) + 1)
            layer_flops = fft_flops + masking_flops
            flops_breakdown['DiffStride'] += layer_flops
            
            current_shape = (H_out, W_out, C)
        
        elif isinstance(layer, tf.keras.layers.Conv2D):
            H, W, C_in = current_shape
            C_out = layer.filters
            k = layer.kernel_size[0]  
            
            kernel_flops = H * W * C_in * C_out * k * k
            flops_breakdown['Conv2D'] += kernel_flops
            
            if layer.use_bias:
                bias_flops = H * W * C_out
                flops_breakdown['BiasAdd'] += bias_flops
            
            if layer.activation is not None and getattr(layer.activation, '__name__', None) != "linear":
                activation_flops = H * W * C_out
                flops_breakdown['Activations'] += activation_flops
            
            current_shape = (H, W, C_out)
            
            if layer == model.output_conv:
                flops_breakdown['OutputConv'] = kernel_flops
                flops_breakdown['Conv2D'] -= kernel_flops
                
                if hasattr(layer.activation, '__name__') and layer.activation.__name__ == 'softmax':
                    softmax_flops = 3 * H * W * layer.filters  
                    flops_breakdown['Softmax'] += softmax_flops
        
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            H, W, C = current_shape
            bn_flops = 4 * H * W * C   
            flops_breakdown['BatchNorm'] += bn_flops
            current_shape = layer.output_shape[1:]
        
        elif isinstance(layer, tf.keras.layers.Concatenate):
            # Assume Concatenate merges along channels (axis=-1)
            # Double the channels (heuristic for U-Net skip connections)
            H, W, C = current_shape
            current_shape = (H, W, C * 2)  # Approximation
            flops_breakdown['Concatenate'] += 0  # No actual FLOPs
        
        elif isinstance(layer, CUFLayer):
            if hasattr(layer, "output_shape"):
                H_target, W_target, out_channels = layer.output_shape[1:]
            else:
                H_target, W_target, out_channels = current_shape
            
            hyper_network = layer.hyper_network
            num_units = hyper_network.num_units  
            dct_basis = hyper_network.dct_basis  
            
            hyper_flops = H_target * W_target * num_units * 3  
            extra_dense_flops = H_target * W_target * num_units * 2  
            
            resize_flops = H_target * W_target * 7  
            patch_filter_flops = H_target * W_target * current_shape[2] * 17
            
            proj_conv_flops = H_target * W_target * (current_shape[2] * layer.filters)
            
            if hasattr(layer.proj_conv, 'use_bias') and layer.proj_conv.use_bias:
                proj_conv_flops += H_target * W_target * layer.filters
            
            if layer.proj_conv.activation is not None and getattr(layer.proj_conv.activation, '__name__', None) != "linear":
                proj_conv_flops += H_target * W_target * layer.filters
            
            total_cuf_flops = hyper_flops + extra_dense_flops + resize_flops + patch_filter_flops + proj_conv_flops
            flops_breakdown['CUF'] += total_cuf_flops
            
            current_shape = (H_target, W_target, layer.filters)
        
        else:
            continue
    
    total_flops = sum(flops_breakdown.values())
    flops_breakdown['Total'] = total_flops
    
    print("\nUpdated FLOP Breakdown:")
    for layer_type, flops in flops_breakdown.items():
        print(f"{layer_type:15}: {flops/1e9:7.2f}G")
    
    return total_flops



class FlopsTracker(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.flops_history = []
        
    def on_train_begin(self, logs=None):
        # Calculate initial FLOPs (untrained model)
        original_eager = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(True)
        flops = get_flops(self.model)
        tf.config.run_functions_eagerly(original_eager)
        self.flops_history.append(flops)  # Epoch 0
        
    def on_epoch_end(self, epoch, logs=None):
        # Calculate FLOPs after training the epoch
        original_eager = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(True)
        flops = get_flops(self.model)
        tf.config.run_functions_eagerly(original_eager)
        self.flops_history.append(flops)  # Epoch 1, 2, ...
        
        # Add to logs
        logs['flops'] = flops
        logs['flops_g'] = flops / 1e9
    
    def on_train_end(self, logs=None):
        # Save full history to model (includes epoch 0)
        self.model.history.history['flops'] = self.flops_history
        self.model.history.history['flops_g'] = [f/1e9 for f in self.flops_history]

