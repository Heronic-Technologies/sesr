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
from typing import Literal

import tensorflow as tf
import tensorflow_datasets as tfds
import tf2onnx
from colorama import Fore, init

from models import model_utils, sesr

init(autoreset=True)

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('epochs', 300, 'Number of epochs to train')
tf.compat.v1.flags.DEFINE_integer('batch_size', 32, 'Batch size during training')
tf.compat.v1.flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate for ADAM')
tf.compat.v1.flags.DEFINE_string('model_name', 'SESR', 'Name of the model')
tf.compat.v1.flags.DEFINE_bool('quant_W', False, 'Quantize weights')
tf.compat.v1.flags.DEFINE_bool('quant_A', False, 'Quantize activations')
tf.compat.v1.flags.DEFINE_bool('gen_tflite', False, 'Generate TFLITE')
tf.compat.v1.flags.DEFINE_integer('tflite_height', 1080, 'Height of LR image in TFLITE')
tf.compat.v1.flags.DEFINE_integer('tflite_width', 1920, 'Width of LR image in TFLITE')
tf.compat.v1.flags.DEFINE_bool('eval_only', False, 'Run validation only (no training)')
tf.compat.v1.flags.DEFINE_string('model_path', '', 'Path to trained model for evaluation')
tf.compat.v1.flags.DEFINE_bool('comb_loss', False, 'Use combined L1 + LPIPS loss instead of just L1')

import utils

#Set some dataset processing parameters and some save/load paths
DATASET_NAME = 'div2k' if FLAGS.scale == 2 else 'div2k/bicubic_x4'
CUSTOM_DATASET = True  #Set to True to use custom dataset instead of DIV2K
DEGRADATION_METHOD: Literal["simple", "bsrgan", "bicubic"] = "bicubic"  #Set degradation method for custom dataset
if CUSTOM_DATASET:
  print(f'{Fore.MAGENTA}Using custom dataset for training and evaluation. Scale: x{FLAGS.scale}, Degradation: {DEGRADATION_METHOD}.')
if not os.path.exists('logs/'):
  os.makedirs('logs/')
BASE_SAVE_DIR = 'logs/x2_models/' if FLAGS.scale == 2 else 'logs/x4_models/'
if not os.path.exists(BASE_SAVE_DIR):
  os.makedirs(BASE_SAVE_DIR)

SUFFIX = 'QAT' if (FLAGS.quant_W and FLAGS.quant_A) else 'FP32'

lpips_loss = utils.LPIPSLoss(net='alex')

if FLAGS.scale == 4: #Specify path to load x2 models (x4 SISR will only finetune x2 models)
  if FLAGS.model_name == 'SESR':
    PATH_2X = 'logs/x2_models/'+FLAGS.model_name+'_m{}_f{}_x2_fs{}_{}_{}_{}Training_{}{}'.format(
                                                                  FLAGS.m,
                                                                  FLAGS.int_features,
                                                                  FLAGS.feature_size,
                                                                  "_relu" if FLAGS.relu_act else '',
                                                                  "_comb" if FLAGS.comb_loss else '',
                                                                  FLAGS.linear_block_type,
                                                                  SUFFIX,
                                                                  f'_custom_{DEGRADATION_METHOD}' if CUSTOM_DATASET else '')

##################################
## TRAINING AND EVALUATION LOOP ##
##################################

def main(unused_argv):

    data_dir = os.getenv("TFDS_DATA_DIR", None)

    if not CUSTOM_DATASET:
        dataset_train, dataset_validation = tfds.load(DATASET_NAME,
                                    split=['train', 'validation'], shuffle_files=True,
                                    data_dir=data_dir)
    else:
        # Option 1: Custom LR/HR pairs
        dataset_train = utils.load_custom_dataset('datasets/DF2K', 'train', lr_folder_suffix=f'{FLAGS.scale}x_{DEGRADATION_METHOD}', lr_file_suffix='x2')
        dataset_validation = utils.load_custom_dataset('datasets/DF2K', 'val', lr_folder_suffix=f'{FLAGS.scale}x_{DEGRADATION_METHOD}', lr_file_suffix='x2')

        # Option 2: Custom HR only (auto-generate LR)
        # dataset_train = utils.load_hr_only_dataset('datasets/hr_only_example_dataset', 'train', scale=FLAGS.scale)
        # dataset_validation = utils.load_hr_only_dataset('datasets/hr_only_example_dataset', 'val', scale=FLAGS.scale)

    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_train = dataset_train.map(utils.rgb_to_y).cache()
    dataset_train = dataset_train.filter(utils.scale_match)
    dataset_train = dataset_train.map(utils.patches).unbatch().shuffle(buffer_size=1_000)

    dataset_validation = dataset_validation.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_validation = dataset_validation.map(utils.rgb_to_y).cache()
    dataset_validation = dataset_validation.filter(utils.scale_match)

    # Set sharding policy to DATA to avoid auto-sharding warnings with custom datasets
    if CUSTOM_DATASET:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset_train = dataset_train.with_options(options)
        dataset_validation = dataset_validation.with_options(options)

    #PSNR metric to be monitored while training.
    def psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.image.psnr(y_true, y_pred, max_val=1.)

    #LPIPS metric to be monitored while training.
    def lpips(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_rgb = utils.y_to_rgb(y_true)
        y_pred_rgb = utils.y_to_rgb(y_pred)
        return lpips_loss(y_true_rgb, y_pred_rgb)

    # Combined loss: L1 + LPIPS
    def combined_loss(y_true, y_pred):
        """
        Recommended: L1 + LPIPS combination
        """
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        lpips_loss = lpips(y_true, y_pred)

        # Standard weights from literature
        total_loss = 1.0 * l1_loss + 0.03 * lpips_loss

        return total_loss

    if FLAGS.eval_only:
        print(f"{Fore.CYAN}Running validation only...")

        model = tf.keras.models.load_model(
            FLAGS.model_path,
            custom_objects={'psnr': psnr, 'lpips': lpips, 'combined_loss': combined_loss}
        )

        results = model.evaluate(
            dataset_validation.batch(1),
            verbose=1
        )

        print(f"{Fore.GREEN}Validation results:")
        for name, value in zip(model.metrics_names, results):
            print(f"  {name}: {value:.4f}")

        return

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # atexit.register(mirrored_strategy._extended._collective_ops._pool.close) # type: ignore (needed for tf2.7?)

    #Select the model to train.
    # with mirrored_strategy.scope():
    if FLAGS.model_name == 'SESR':
      if FLAGS.linear_block_type=='collapsed':
        LinearBlock_fn = model_utils.LinearBlock_c
      else:
        LinearBlock_fn = model_utils.LinearBlock_e
      model = sesr.SESR(
        m=FLAGS.m,
        feature_size=FLAGS.feature_size,
        LinearBlock_fn=LinearBlock_fn,
        quant_W=FLAGS.quant_W > 0,
        quant_A=FLAGS.quant_A > 0,
        gen_tflite = FLAGS.gen_tflite,
        mode='train')

    #Declare the optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate,
                                      amsgrad=True)

    #If scale == 4, base x2 model must be loaded for transfer learning.
    #Load the pretrained weights into the base model from x2 SISR:
    if FLAGS.scale == 4:
      if CUSTOM_DATASET and os.path.exists(FLAGS.model_path):
        print(f"{Fore.CYAN}Loading model from {FLAGS.model_path} for finetuning on custom dataset...")
        model = tf.keras.models.load_model(
            FLAGS.model_path,
            custom_objects={'psnr': psnr, 'lpips': lpips, 'combined_loss': combined_loss}
        )
      else:
        print(f"{Fore.CYAN}Loading x2 model from {PATH_2X} for training the x4 model...")
        base_model = tf.keras.models.load_model(PATH_2X, custom_objects={'psnr': psnr, 'lpips': lpips, 'combined_loss': combined_loss})
        layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
        for layer in model.layers:
          layer_name = layer.name
          if FLAGS.model_name == 'SESR':
            if layer_name != 'linear_block_{}'.format(FLAGS.m+1): #Last layer in x4 is not the same as that in x2 for SESR
              print(layer_name)
              layer.set_weights = layer_dict[layer_name].get_weights()
    if FLAGS.scale == 2:
      if CUSTOM_DATASET and os.path.exists(FLAGS.model_path):
        print(f"{Fore.CYAN}Loading model from {FLAGS.model_path} for finetuning on custom dataset...")
        model = tf.keras.models.load_model(
            FLAGS.model_path,
            custom_objects={'psnr': psnr, 'lpips': lpips, 'combined_loss': combined_loss}
        )

    #Compile and train the model.
    if FLAGS.comb_loss:
        print(f"{Fore.CYAN}Using combined L1 + LPIPS loss for training.")
        model.compile(optimizer=optimizer, loss=combined_loss, metrics=[psnr, lpips])
    else:
        print(f"{Fore.CYAN}Using L1 loss for training.")
        model.compile(optimizer=optimizer, loss='mae', metrics=[psnr, lpips])

    # End of mirrored_strategy.scope()

    # for lr_y, hr_y in dataset_train:
        # print(f"{Fore.CYAN}Sample LR patch shape: {lr_y.shape} ({lr_y.dtype}), Sample HR patch shape: {hr_y.shape} ({hr_y.dtype})")
        # print(f"{Fore.CYAN}LR pixel value range: [{tf.reduce_min(lr_y).numpy():.4f}, {tf.reduce_max(lr_y).numpy():.4f}], HR pixel value range: [{tf.reduce_min(hr_y).numpy():.4f}, {tf.reduce_max(hr_y).numpy():.4f}]")
        # model_out = model(lr_y[None, ...], training=False)
        # print(f"{Fore.CYAN}Model output shape: {model_out.shape}, Model output dtype: {model_out.dtype}, Model output pixel value range: [{tf.reduce_min(model_out).numpy():.4f}, {tf.reduce_max(model_out).numpy():.4f}]")
        # print(f"{Fore.CYAN}HR shape: {hr_y[None, ...].shape}, HR dtype: {hr_y[None, ...].dtype}, HR pixel value range: [{tf.reduce_min(hr_y[None, ...]).numpy():.4f}, {tf.reduce_max(hr_y[None, ...]).numpy():.4f}]")
        # lpips_value = lpips(hr_y[None, ...], model_out)
        # print(f"{Fore.CYAN}LPIPS value for sample patch: {lpips_value:.4f}")
    # exit()

    model.fit(dataset_train.batch(FLAGS.batch_size),
              epochs=FLAGS.epochs,
              validation_data=dataset_validation.batch(1),
              validation_freq=10)
    model.summary()

    #Save the trained models.
    if FLAGS.model_name == 'SESR':
      final_save_path = BASE_SAVE_DIR+FLAGS.model_name+'_m{}_f{}_x{}_fs{}{}{}_{}Training_{}{}'.format(
                           FLAGS.m, FLAGS.int_features, FLAGS.scale, FLAGS.feature_size, "_relu" if FLAGS.relu_act else '', "_comb" if FLAGS.comb_loss else '',
                           FLAGS.linear_block_type, SUFFIX, f'_custom_{DEGRADATION_METHOD}' if CUSTOM_DATASET else '')
      model.save(final_save_path)
      model.save_weights(final_save_path + '/model_weights')

      # convert to ONNX
      spec = [tf.TensorSpec((1, None, None, 1), tf.float32, name="input_1")]
      output_path = final_save_path + '/model.onnx'
      tf2onnx.convert.from_keras(model, input_signature=spec, opset=16,
                                      output_path=output_path, inputs_as_nchw=["input_1"],
                                      outputs_as_nchw=["output_1"])



      #Get the TFLITE for custom image size
      if FLAGS.gen_tflite:
        y_lr = tf.random.uniform([1, FLAGS.tflite_height, FLAGS.tflite_width, 1],
                                minval=0., maxval=1.)
        model_tflite = sesr.SESR(
          m=FLAGS.m,
          feature_size=FLAGS.feature_size,
          LinearBlock_fn=LinearBlock_fn,
          quant_W=FLAGS.quant_W > 0,
          quant_A=FLAGS.quant_A > 0,
          gen_tflite = FLAGS.gen_tflite,
          mode='infer')
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate,
                                      amsgrad=True)
        model_tflite.load_weights(final_save_path + '/model_weights')
        if FLAGS.comb_loss:
            print(f"{Fore.CYAN}Compiling TFLITE model with combined L1 + LPIPS loss for quantization...")
            model_tflite.compile(optimizer=optimizer, loss=combined_loss, metrics=[psnr, lpips])
        else:
            print(f"{Fore.CYAN}Compiling TFLITE model with L1 loss for quantization...")
            model_tflite.compile(optimizer=optimizer, loss='mae', metrics=[psnr, lpips])
        # build (execute forward pass)
        model_tflite(y_lr)
        utils.generate_int8_tflite(
          model_tflite,
          'model_quantized',
          final_save_path,
          fake_quant=True)


if __name__ == '__main__':
    tf.compat.v1.app.run()
