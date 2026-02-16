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

# OPTIMIZATION FLAGS
tf.compat.v1.flags.DEFINE_bool('use_mixed_precision', False, 'Use mixed precision training for faster computation')
tf.compat.v1.flags.DEFINE_bool('skip_lpips_metric', True, 'Skip LPIPS metric during training (only compute on validation)')

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

if FLAGS.scale == 4: #Specify path to load x2 models (x4 SISR will only finetune x2 models)
  if FLAGS.model_name == 'SESR':
    PATH_2X = 'logs/x2_models/'+FLAGS.model_name+'_m{}_f{}_x2_fs{}{}{}_{}Training_{}{}'.format(
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
    # Initialize metrics and losses with optimizations
    lpips_loss = utils.LPIPSLoss(
        net='mobilenetv2',
        use_mixed_precision=FLAGS.use_mixed_precision
    )
    lpips_weight_var = tf.Variable(0.0, trainable=False)

    lpips_metric = None
    if not FLAGS.skip_lpips_metric:
        print(f"{Fore.YELLOW}Warning: Computing LPIPS metric during training will slow down training significantly!")
        print(f"{Fore.YELLOW}Consider setting --skip_lpips_metric=True to only compute it on validation.")
        lpips_metric = utils.LPIPSMetric(net='alex')

    # Enable mixed precision if requested
    if FLAGS.use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    data_dir = os.getenv("TFDS_DATA_DIR", None)

    if not CUSTOM_DATASET:
        dataset_train, dataset_validation = tfds.load(DATASET_NAME,
                                    split=['train', 'validation'], shuffle_files=True,
                                    data_dir=data_dir)
    else:
        dataset_train = utils.load_custom_dataset('datasets/DF2K', 'train', lr_folder_suffix=f'{FLAGS.scale}x_{DEGRADATION_METHOD}', lr_file_suffix=f'x{FLAGS.scale}')
        dataset_validation = utils.load_custom_dataset('datasets/DF2K', 'val', lr_folder_suffix=f'{FLAGS.scale}x_{DEGRADATION_METHOD}', lr_file_suffix=f'x{FLAGS.scale}')

        # Option 2: Custom HR only (auto-generate LR)
        # dataset_train = utils.load_hr_only_dataset('datasets/hr_only_example_dataset', 'train', scale=FLAGS.scale)
        # dataset_validation = utils.load_hr_only_dataset('datasets/hr_only_example_dataset', 'val', scale=FLAGS.scale)

    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_train = dataset_train.map(utils.rgb_to_y, num_parallel_calls=tf.data.AUTOTUNE).cache()
    dataset_train = dataset_train.filter(utils.scale_match)
    dataset_train = dataset_train.map(utils.patches, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_train = dataset_train.unbatch().shuffle(buffer_size=1_000)

    dataset_validation = dataset_validation.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_validation = dataset_validation.map(utils.rgb_to_y, num_parallel_calls=tf.data.AUTOTUNE).cache()
    dataset_validation = dataset_validation.filter(utils.scale_match)

    # Set sharding policy to DATA to avoid auto-sharding warnings with custom datasets
    if CUSTOM_DATASET:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset_train = dataset_train.with_options(options)
        dataset_validation = dataset_validation.with_options(options)

    # Define metrics and losses.
    @tf.function
    def psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.image.psnr(y_true, y_pred, max_val=1.)

    # LPIPS metric wrapper (only used if not skipped).
    def lpips(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if FLAGS.skip_lpips_metric:
            return tf.constant("N/A", dtype=tf.string)  # Return dummy value during training
        y_true_rgb = utils.y_to_rgb(y_true)
        y_pred_rgb = utils.y_to_rgb(y_pred)
        return lpips_metric(y_true_rgb, y_pred_rgb)

    @tf.function
    def l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss =  tf.reduce_mean(tf.abs(y_true - y_pred))
        return tf.cast(loss, tf.float32)

    # Perceptual loss is always computed via TF
    def perceptual_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = lpips_loss(y_true, y_pred)
        return tf.cast(loss, tf.float32)

    # Combined loss: L1 + LPIPS
    @tf.function
    def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        l1 = l1_loss(y_true, y_pred)
        p = perceptual_loss(y_true, y_pred)
        l1 = tf.cast(l1, tf.float32)
        p = tf.cast(p, tf.float32)
        return l1 + lpips_weight_var * p

    if FLAGS.eval_only:
        print(f"{Fore.CYAN}Running validation only...")

        # For evaluation, always compute LPIPS metric
        if lpips_metric is None:
            lpips_metric = utils.LPIPSMetric(net='alex')

        def lpips_eval(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y_true_rgb = utils.y_to_rgb(y_true)
            y_pred_rgb = utils.y_to_rgb(y_pred)
            return lpips_metric(y_true_rgb, y_pred_rgb)

        model = tf.keras.models.load_model(
            FLAGS.model_path,
            custom_objects={'psnr': psnr, 'lpips': lpips_eval, 'combined_loss': combined_loss}
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

    # Declare the optimizer.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=FLAGS.learning_rate,
        amsgrad=True
    )

    # Use loss scaling for mixed precision.
    if FLAGS.use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # If scale == 4, base x2 model must be loaded for transfer learning
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

    # Compile the model.
    if FLAGS.comb_loss:
        print(f"{Fore.CYAN}Using combined L1 + LPIPS loss for training.")
        if FLAGS.skip_lpips_metric:
            compile_metrics = [l1_loss, perceptual_loss, psnr]
        else:
            compile_metrics = [l1_loss, perceptual_loss, psnr, lpips]

        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=compile_metrics,
        )
    else:
        print(f"{Fore.CYAN}Using L1 loss for training.")
        if FLAGS.skip_lpips_metric:
            compile_metrics = [l1_loss, perceptual_loss, psnr]
        else:
            compile_metrics = [l1_loss, perceptual_loss, psnr, lpips]

        model.compile(
            optimizer=optimizer,
            loss='mae',
            metrics=compile_metrics,
        )

    class ValidationLPIPSCallback(tf.keras.callbacks.Callback):
        def __init__(self, validation_data, metric_fn):
            super().__init__()
            self.validation_data = validation_data
            self.metric_fn = metric_fn

        def on_epoch_end(self, epoch, logs=None):
            if self.metric_fn is not None:
                # Compute LPIPS on validation set
                lpips_values = []
                for lr, hr in self.validation_data.take(10):  # Sample 10 images
                    pred = self.model(lr, training=False)
                    hr_rgb = utils.y_to_rgb(hr)
                    pred_rgb = utils.y_to_rgb(pred)
                    lpips_val = self.metric_fn(hr_rgb, pred_rgb)
                    lpips_values.append(float(lpips_val))

                avg_lpips = sum(lpips_values) / len(lpips_values)
                logs['val_lpips'] = avg_lpips
                print(f"\n{Fore.CYAN}Validation LPIPS: {avg_lpips:.4f}")

    class AdaptiveLPIPSScheduler(tf.keras.callbacks.Callback):
        def __init__(self, start_weight=0.0, end_weight=0.05, ramp_epochs=10):
            super().__init__()
            self.start_weight = start_weight
            self.end_weight = end_weight
            self.ramp_epochs = ramp_epochs

        def on_epoch_begin(self, epoch, logs=None):
            if epoch < self.ramp_epochs:
                progress = epoch / self.ramp_epochs
                current_weight = self.start_weight + (self.end_weight - self.start_weight) * progress
            else:
                current_weight = self.end_weight

            lpips_weight_var.assign(current_weight)

            print(f"\nEpoch {epoch + 1} - LPIPS weight: {current_weight:.6f}")


    callbacks = []

    # if FLAGS.skip_lpips_metric:
    #     if lpips_metric is None:
    #         print(f"{Fore.YELLOW}Initializing LPIPS metric for validation...")
    #         lpips_metric = utils.LPIPSMetric(net='alex')
    #     callbacks.append(ValidationLPIPSCallback(dataset_validation.batch(1), lpips_metric))

    log_dir = os.path.join("tensorboard_logs", "{}_m{}_f{}_x{}_fs{}{}{}_{}Training_{}{}".format(FLAGS.model_name, FLAGS.m, FLAGS.int_features, FLAGS.scale, FLAGS.feature_size, "_relu" if FLAGS.relu_act else '', "_comb" if FLAGS.comb_loss else '', FLAGS.linear_block_type, SUFFIX, f'_custom_{DEGRADATION_METHOD}' if CUSTOM_DATASET else ''))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        update_freq='epoch',
    )

    callbacks.append(tensorboard_callback)
    callbacks.append(AdaptiveLPIPSScheduler(start_weight=0.0, end_weight=0.05, ramp_epochs=10))

    # Train the model
    print(f"{Fore.GREEN}Starting training with optimizations:")
    print(f"  - Mixed precision: {FLAGS.use_mixed_precision}")
    print(f"  - Skip LPIPS during training: {FLAGS.skip_lpips_metric}")
    print(f"  - Batch size: {FLAGS.batch_size}")

    model.fit(
        dataset_train.batch(FLAGS.batch_size),
        epochs=FLAGS.epochs,
        validation_data=dataset_validation.batch(1),
        validation_freq=5,
        callbacks=callbacks,
    )
    model.summary()

    # Save the trained models
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

      # Get the TFLITE for custom image size
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
