# Copyright 2021 Arm Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
from pathlib import Path
from typing import List, Optional, Tuple

import lpips
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from colorama import Fore, init

init(autoreset=True)

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('scale', 2, 'Scale of SISR')

#Set some dataset related parameters
SCALE = FLAGS.scale
if SCALE != 2 and SCALE != 4:
  raise ValueError('Only x2 or x4 SISR is currently supported')
# PATCH_SIZE_HR = 128 if SCALE == 2 else 200
PATCH_SIZE_HR = 128 if SCALE == 2 else 256
# PATCH_SIZE_HR = 128 if SCALE == 2 else 384
PATCH_SIZE_LR = PATCH_SIZE_HR // SCALE
PATCHES_PER_IMAGE = 64


###########################
## DATASET PREPROCESSING ##
###########################


#Convert RGB image to YCbCr
def rgb_to_ycbcr(rgb: tf.Tensor) -> tf.Tensor:
    ycbcr_from_rgb = tf.constant([[65.481, 128.553, 24.966],
                                  [-37.797, -74.203, 112.0],
                                  [112.0, -93.786, -18.214]])
    rgb = tf.cast(rgb, dtype=tf.dtypes.float32) / 255.
    ycbcr = tf.linalg.matmul(rgb, ycbcr_from_rgb, transpose_b=True)
    return ycbcr + tf.constant([[[16., 128., 128.]]])

#Convert YCbCr image to RGB
def ycbcr_to_rgb(ycbcr: tf.Tensor) -> tf.Tensor:
    rgb_from_ycbcr = tf.constant([[0.00456621, 0.00456621, 0.00456621],
                                  [0, -0.00153632, 0.00791071],
                                  [0.00625893, -0.00318811, 0]])

    ycbcr = ycbcr - tf.constant([[[16., 128., 128.]]])
    rgb = tf.linalg.matmul(ycbcr, rgb_from_ycbcr, transpose_b=True)
    rgb = tf.clip_by_value(rgb, 0., 1.)
    return rgb

#Get the Y-Channel only
def rgb_to_y(example: tfds.features.FeaturesDict) -> Tuple[tf.Tensor, tf.Tensor]:
    lr_ycbcr = rgb_to_ycbcr(example['lr'])
    hr_ycbcr = rgb_to_ycbcr(example['hr'])
    return lr_ycbcr[..., 0:1] / 255., hr_ycbcr[..., 0:1] / 255.


#Convert Y-Channel to RGB (replicate Y across 3 channels)
def y_to_rgb(y: tf.Tensor) -> tf.Tensor:
    rgb = tf.concat([y, y, y], axis=-1)
    return rgb

#Extract random patches for training
def random_patch(lr: tf.Tensor, hr: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    def lr_offset(axis: int):
        size = tf.shape(lr)[axis]
        return tf.random.uniform(shape=(), maxval=size - PATCH_SIZE_LR,
                                 dtype=tf.dtypes.int32)

    lr_offset_x, lr_offset_y = lr_offset(axis=0), lr_offset(axis=1)
    hr_offset_x, hr_offset_y = SCALE * lr_offset_x, SCALE * lr_offset_y
    lr = lr[lr_offset_x:lr_offset_x + PATCH_SIZE_LR,
            lr_offset_y:lr_offset_y + PATCH_SIZE_LR]
    hr = hr[hr_offset_x:hr_offset_x + PATCH_SIZE_HR,
            hr_offset_y:hr_offset_y + PATCH_SIZE_HR]
    return lr, hr


#Data augmentions
def augment(lr: tf.Tensor, hr: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    u = tf.random.uniform(shape=())
    k = tf.random.uniform(shape=(), maxval=4, dtype=tf.dtypes.int32)

    def augment_(image: tf.Tensor) -> tf.Tensor:
        image = tf.cond(u < 0.5, true_fn=lambda: image, false_fn=lambda: tf.image.flip_up_down(image))
        return tf.image.rot90(image, k=k)

    return augment_(lr), augment_(hr)


#Get many random patches for each image
def patches(lr: tf.Tensor, hr: tf.Tensor) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    tuples = (augment(*random_patch(lr, hr)) for _ in range(PATCHES_PER_IMAGE))
    lr, hr = zip(*tuples)
    return list(lr), list(hr)

def scale_match(lr, hr):
    lr_shape = tf.shape(lr)  # [H, W, C]
    hr_shape = tf.shape(hr)

    # Basic sanity: rank 3
    ok_rank = tf.logical_and(tf.equal(tf.rank(lr), 3), tf.equal(tf.rank(hr), 3))

    # Positive dims (avoid rot90 assertion / empty tensors)
    ok_pos = tf.reduce_all(lr_shape[:3] > 0) & tf.reduce_all(hr_shape[:3] > 0)

    # Channels match
    ok_c = tf.equal(lr_shape[2], hr_shape[2])

    # Scale match
    ok_scale = tf.logical_and(
        tf.equal(hr_shape[0], lr_shape[0] * FLAGS.scale),
        tf.equal(hr_shape[1], lr_shape[1] * FLAGS.scale),
    )

    # Optional: skip “tiny” images
    ok_min = tf.logical_and(lr_shape[0] > 2, lr_shape[1] > 2) & tf.logical_and(hr_shape[0] > 2, hr_shape[1] > 2)

    return ok_rank & ok_pos & ok_c & ok_scale & ok_min

#Generate INT8 TFLITE
def generate_int8_tflite(model: tf.keras.Model,
                         filename: str,
                         path: Optional[str] = '/tmp',
                         fake_quant: bool = False) -> str:
    saved_model = path + '/' + filename
    model.save(saved_model)
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model)
    converter.inference_type = tf.dtypes.int8
    if fake_quant:  # give some default ranges for activations (for perf-eval only)
        converter.default_ranges_stats = (-6., 6.)
    input_arrays = converter.get_input_arrays()
    # if input node has fake-quant node, then the following ranges would be
    # overridden by fake-quant ranges.
    converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    if not os.path.exists(path):
        os.makedirs(path)
    tflite_filename = path + '/' + filename + '.tflite'
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)

    return tflite_filename

def load_custom_dataset(data_dir: str, split: str = 'train', hr_folder_suffix: str = '', lr_folder_suffix: str = '', hr_file_suffix: str = '', lr_file_suffix: str = '') -> tf.data.Dataset:
    """Load a custom SR dataset from folder structure."""
    print(f"{Fore.CYAN}Loading dataset from {data_dir}, split: {split}, hr_suffix: {hr_folder_suffix}, lr_suffix: {lr_folder_suffix}")

    lr_dir = os.path.join(data_dir, split, 'lr', lr_folder_suffix)
    hr_dir = os.path.join(data_dir, split, 'hr', hr_folder_suffix)

    hr_files = []
    lr_files = []

    for hr_file in sorted(tf.io.gfile.glob(os.path.join(hr_dir, '*'))):
        hr_filename = os.path.basename(hr_file)
        if hr_file_suffix != '':
            lr_filename = hr_filename.replace(hr_file_suffix, lr_file_suffix)
        else:
            lr_filename = Path(hr_file).stem + lr_file_suffix + Path(hr_file).suffix
        lr_path = os.path.join(lr_dir, lr_filename)
        if tf.io.gfile.exists(lr_path):
            lr_files.append(lr_path)
            hr_files.append(hr_file)
        else:
            print(f"{Fore.RED}WARNING: LR file not found for HR file {hr_file}: expected {lr_path}. This pair will be skipped.")

    if len(lr_files) != len(hr_files):
        print(f"{Fore.RED}WARNING: Mismatch in number of LR and HR images!")
        exit()

    def load_image_pair(lr_path, hr_path):
        lr = tf.io.read_file(lr_path)
        lr = tf.image.decode_image(lr, channels=3, expand_animations=False)
        hr = tf.io.read_file(hr_path)
        hr = tf.image.decode_image(hr, channels=3, expand_animations=False)
        return {'lr': lr, 'hr': hr}

    dataset = tf.data.Dataset.from_tensor_slices((lr_files, hr_files))
    dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def load_hr_only_dataset(data_dir: str, split: str = 'train', scale: int = 2, hr_suffix: str = '') -> tf.data.Dataset:
    """Load dataset from HR images only, generating LR via bicubic downsampling."""
    print(f"{Fore.CYAN}Loading HR-only dataset from {data_dir}, split: {split}, scale: {scale}, hr_suffix: {hr_suffix}")

    hr_dir = os.path.join(data_dir, split, 'hr', hr_suffix)
    hr_files = sorted(tf.io.gfile.glob(os.path.join(hr_dir, '*')))

    def load_and_downsample(hr_path):
        hr = tf.io.read_file(hr_path)
        hr = tf.image.decode_image(hr, channels=3, expand_animations=False)
        hr = tf.cast(hr, tf.float32)

        # Get dimensions and ensure divisibility by scale
        shape = tf.shape(hr)
        h = (shape[0] // scale) * scale
        w = (shape[1] // scale) * scale
        hr = hr[:h, :w, :]

        # Downsample to create LR
        lr = tf.image.resize(hr, [h // scale, w // scale], method='bicubic')
        lr = tf.clip_by_value(lr, 0, 255)

        return {'lr': tf.cast(lr, tf.uint8), 'hr': tf.cast(hr, tf.uint8)}

    dataset = tf.data.Dataset.from_tensor_slices(hr_files)
    dataset = dataset.map(load_and_downsample, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

class LPIPSLoss:
    """
    Differentiable LPIPS-like perceptual loss in TensorFlow.

    Inputs:
        y_true, y_pred: TF tensors [B,H,W,1] or [B,H,W,3] in [0, 1]

    Nets:
        - net='vgg'  -> VGG16 (ImageNet) feature loss (classic perceptual loss)
        - net='mobilenetv2' -> MobileNetV2 (ImageNet) feature loss (acts as an "Alex-like" lightweight backbone)
    """

    def __init__(self, net='mobilenetv2', use_gpu=True,
                 layer_weights=None,
                 eps=1e-10):
        self.net = net.lower()
        self.eps = tf.constant(eps, tf.float32)

        if self.net == 'vgg':
            layer_names = [
                "block1_conv2",
                "block2_conv2",
                "block3_conv3",
                "block4_conv3",
                "block5_conv3",
            ]
            if layer_weights is None:
                layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
            self._build_vgg16(layer_names)

        elif self.net == 'mobilenetv2':
            layer_names = [
                "block_1_expand_relu",
                "block_3_expand_relu",
                "block_6_expand_relu",
                "block_13_expand_relu",
                "out_relu",
            ]
            if layer_weights is None:
                layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
            self._build_mobilenetv2(layer_names)

        else:
            raise ValueError("net must be 'mobilenetv2' or 'vgg'")

        self.layer_weights = tf.constant(layer_weights, dtype=tf.float32)

    # ---------- Backbone builders ----------
    def _build_vgg16(self, layer_names):
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        base.trainable = False
        outs = [base.get_layer(n).output for n in layer_names]
        self.feature_model = tf.keras.Model(base.input, outs, name="vgg16_features")
        self.feature_model.trainable = False
        self._preprocess = self._preprocess_vgg

    def _build_mobilenetv2(self, layer_names):
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet")
        base.trainable = False
        outs = [base.get_layer(n).output for n in layer_names]
        self.feature_model = tf.keras.Model(base.input, outs, name="mobilenetv2_features")
        self.feature_model.trainable = False
        self._preprocess = self._preprocess_mobilenetv2

    # ---------- Input helpers ----------
    def _preprocess_vgg(self, x_01_rgb):
        """
        VGG16 preprocess:
          expects RGB in 0..255 and does the standard vgg16.preprocess_input transform.
        """
        x = tf.clip_by_value(x_01_rgb, 0.0, 1.0) * 255.0
        return tf.keras.applications.vgg16.preprocess_input(x)

    def _preprocess_mobilenetv2(self, x_01_rgb):
        """
        MobileNetV2 preprocess:
          expects RGB in 0..255 then maps to [-1,1] internally.
        This is close to LPIPS-style input scaling.
        """
        x = tf.clip_by_value(x_01_rgb, 0.0, 1.0) * 255.0
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)

    @staticmethod
    def _channelwise_unit_normalize(f, eps):
        """
        LPIPS-style-ish feature normalization:
        normalize each spatial position by channel norm.
        f: [B,H,W,C]
        """
        denom = tf.sqrt(tf.reduce_sum(tf.square(f), axis=-1, keepdims=True) + eps)
        return f / denom

    # ---------- Main call ----------
    def __call__(self, y_true, y_pred):
        """
        Returns:
            TF scalar perceptual distance (differentiable w.r.t y_pred)
        """
        # 1) Y -> RGB
        y_true_rgb = y_to_rgb(y_true)
        y_pred_rgb = y_to_rgb(y_pred)

        # 2) Backbone-specific preprocessing
        y_true_in = self._preprocess(y_true_rgb)
        y_pred_in = self._preprocess(y_pred_rgb)

        # 3) Extract features
        ft_list = self.feature_model(y_true_in, training=False)
        fp_list = self.feature_model(y_pred_in, training=False)
        if not isinstance(ft_list, (list, tuple)):
            ft_list, fp_list = [ft_list], [fp_list]

        # 4) LPIPS-like distance: sum over layers of feature differences
        #    Using L2 on channel-normalized features (stable for SR).
        losses = []
        for w, ft, fp in zip(tf.unstack(self.layer_weights), ft_list, fp_list):
            ft_n = self._channelwise_unit_normalize(ft, self.eps)
            fp_n = self._channelwise_unit_normalize(fp, self.eps)
            # mean over batch + spatial + channels
            losses.append(w * tf.reduce_mean(tf.square(ft_n - fp_n)))

        return tf.add_n(losses)

class LPIPSMetric:
    """
    PyTorch LPIPS wrapper for TensorFlow/Y-channel.
    """

    def __init__(self, net='alex', use_gpu=True):
        """
        Args:
            net: 'alex', 'vgg', or 'squeeze'
            use_gpu: Use GPU if available
        """
        self.net = net
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

        print(f"Initializing LPIPS ({net}) on {self.device}")

        # Initialize LPIPS model
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.loss_fn.eval()  # Set to evaluation mode

    def _compute_lpips_numpy(self, y_true_np, y_pred_np):
        """
        Compute LPIPS between Y channel images.

        Args:
            y_true: [B, H, W, 1] Y channel ground truth in [0, 1]
            y_pred: [B, H, W, 1] Y channel prediction in [0, 1]

        Returns:
            LPIPS distance (scalar, typically 0.0-1.0)
        """
        # Convert Y [B, H, W, 1] to grayscale RGB [B, H, W, 3]
        if y_true_np.shape[-1] == 1:
            y_true_rgb = np.concatenate([y_true_np, y_true_np, y_true_np], axis=-1)
        elif y_true_np.shape[-1] == 3:
            y_true_rgb = y_true_np
        else:
            raise ValueError(f"Expected y_true to have 1 or 3 channels, got {y_true_np.shape[-1]}")
        if y_pred_np.shape[-1] == 1:
            y_pred_rgb = np.concatenate([y_pred_np, y_pred_np, y_pred_np], axis=-1)
        elif y_pred_np.shape[-1] == 3:
            y_pred_rgb = y_pred_np
        else:
            raise ValueError(f"Expected y_pred to have 1 or 3 channels, got {y_pred_np.shape[-1]}")

        # Convert to PyTorch format: [B, H, W, C] -> [B, C, H, W]
        y_true_rgb = np.transpose(y_true_rgb, (0, 3, 1, 2))
        y_pred_rgb = np.transpose(y_pred_rgb, (0, 3, 1, 2))

        # Convert to PyTorch tensors in [-1, 1] range
        y_true_torch = torch.from_numpy(y_true_rgb).float().to(self.device)
        y_pred_torch = torch.from_numpy(y_pred_rgb).float().to(self.device)

        # Scale from [0, 1] to [-1, 1]
        y_true_torch = y_true_torch * 2.0 - 1.0
        y_pred_torch = y_pred_torch * 2.0 - 1.0

        # Compute LPIPS
        with torch.no_grad():
            distance = self.loss_fn(y_true_torch, y_pred_torch)

        # Return mean distance as numpy float
        return np.float32(distance.mean().cpu().numpy())

    def __call__(self, y_true, y_pred):
        """
        Compute LPIPS - works in both eager and graph mode.

        Args:
            y_true: TF tensor [B, H, W, 1] in [0, 1]
            y_pred: TF tensor [B, H, W, 1] in [0, 1]

        Returns:
            TF scalar: LPIPS distance
        """
        # Use tf.py_function to call PyTorch code
        lpips_value = tf.py_function(
            func=self._compute_lpips_numpy,
            inp=[y_true, y_pred],
            Tout=tf.float32
        )

        # Ensure it's a scalar
        lpips_value = tf.reshape(lpips_value, [])

        return tf.stop_gradient(lpips_value)