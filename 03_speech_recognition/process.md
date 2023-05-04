 2023-04-25 10:02:49.341611: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-04-25 10:02:49.341645: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
/home/eason/.local/lib/python3.8/site-packages/tensorflow_addons/utils/tfa_eol_msg.py:23: UserWarning: 

TensorFlow Addons (TFA) has ended development and introduction of new features.
TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP). 

For more information see: https://github.com/tensorflow/addons/issues/2807 

  warnings.warn(
/home/eason/.local/lib/python3.8/site-packages/tensorflow_addons/utils/ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.10.0 and strictly below 2.13.0 (nightly versions are not supported). 
 The versions of TensorFlow you are currently using is 2.8.4 and is not supported. 
Some things might work, some things might not.
If you were to encounter a bug, do not file an issue.
If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
You can find the compatibility matrix in TensorFlow Addon's readme:
https://github.com/tensorflow/addons
  warnings.warn(
TensorFlow Version: 2.8.4
Model Maker Version: 0.4.2
======================== <01> 导入所需的包 ========================
Splitting exercise_bike.wav
Splitting dude_miaowing.wav
Splitting running_tap.wav
Splitting pink_noise.wav
Splitting white_noise.wav
Splitting doing_the_dishes.wav
Splitting throat_clearing.wav
Splitting silence.wav
======================== <02.1> 准备数据集：生成背景噪声数据集 ========================
======================== <02.2> 准备数据集：准备语音命令数据集 ========================
======================== <02.3> 准备数据集：准备自定义数据集 ========================
Class: left
File: dataset-speech/left/023a61ad_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <02.4> 准备数据集：播放样本 ========================
WARNING:tensorflow:SavedModel saved prior to TF 2.5 detected when loading Keras model. Please ensure that you are saving the model with model.save() or tf.keras.models.save_model(), *NOT* tf.saved_model.save(). To confirm, there should be a file named "keras_metadata.pb" in the SavedModel directory.
2023-04-25 10:02:55.639909: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/eason/.local/lib/python3.8/site-packages/cv2/../../lib64:
2023-04-25 10:02:55.639995: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2023-04-25 10:02:55.640057: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (Arch): /proc/driver/nvidia/version does not exist
2023-04-25 10:02:55.640793: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
======================== <03> 定义模型 ========================
======================== <04> 加载数据集 ========================
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_1 (Conv2D)           (None, 42, 225, 8)        136       
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 21, 112, 8)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 20, 109, 32)       2080      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 54, 32)       0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 9, 51, 32)         8224      
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 4, 25, 32)        0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 3, 22, 32)         8224      
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 11, 32)        0         
 2D)                                                             
                                                                 
 flatten_1 (Flatten)         (None, 704)               0         
                                                                 
 dropout_1 (Dropout)         (None, 704)               0         
                                                                 
 dense_1 (Dense)             (None, 2000)              1410000   
                                                                 
 dropout_2 (Dropout)         (None, 2000)              0         
                                                                 
 classification_head (Dense)  (None, 9)                18009     
                                                                 
=================================================================
Total params: 1,446,673
Trainable params: 18,009
Non-trainable params: 1,428,664
_________________________________________________________________
Epoch 1/30
40/40 [==============================] - 89s 2s/step - loss: 2.0710 - acc: 0.4811 - val_loss: 0.4493 - val_acc: 0.8583
Epoch 2/30
40/40 [==============================] - 1s 30ms/step - loss: 0.6680 - acc: 0.7840 - val_loss: 0.2939 - val_acc: 0.9134
Epoch 3/30
40/40 [==============================] - 1s 30ms/step - loss: 0.4512 - acc: 0.8567 - val_loss: 0.2433 - val_acc: 0.9449
Epoch 4/30
40/40 [==============================] - 1s 34ms/step - loss: 0.3355 - acc: 0.8905 - val_loss: 0.2258 - val_acc: 0.9291
Epoch 5/30
40/40 [==============================] - 1s 31ms/step - loss: 0.2878 - acc: 0.9089 - val_loss: 0.2128 - val_acc: 0.9370
Epoch 6/30
40/40 [==============================] - 1s 30ms/step - loss: 0.2691 - acc: 0.9120 - val_loss: 0.2148 - val_acc: 0.9488
Epoch 7/30
40/40 [==============================] - 1s 30ms/step - loss: 0.2050 - acc: 0.9243 - val_loss: 0.2076 - val_acc: 0.9528
Epoch 8/30
40/40 [==============================] - 1s 30ms/step - loss: 0.2101 - acc: 0.9284 - val_loss: 0.2016 - val_acc: 0.9528
Epoch 9/30
40/40 [==============================] - 1s 30ms/step - loss: 0.1862 - acc: 0.9406 - val_loss: 0.1951 - val_acc: 0.9528
Epoch 10/30
40/40 [==============================] - 1s 30ms/step - loss: 0.1891 - acc: 0.9324 - val_loss: 0.2070 - val_acc: 0.9409
Epoch 11/30
40/40 [==============================] - 1s 30ms/step - loss: 0.1537 - acc: 0.9478 - val_loss: 0.2084 - val_acc: 0.9528
Epoch 12/30
40/40 [==============================] - 1s 31ms/step - loss: 0.1321 - acc: 0.9509 - val_loss: 0.2092 - val_acc: 0.9567
Epoch 13/30
40/40 [==============================] - 1s 30ms/step - loss: 0.1141 - acc: 0.9611 - val_loss: 0.1944 - val_acc: 0.9528
Epoch 14/30
40/40 [==============================] - 1s 30ms/step - loss: 0.1273 - acc: 0.9570 - val_loss: 0.1891 - val_acc: 0.9528
Epoch 15/30
40/40 [==============================] - 1s 30ms/step - loss: 0.1237 - acc: 0.9539 - val_loss: 0.2051 - val_acc: 0.9488
Epoch 16/30
40/40 [==============================] - 1s 31ms/step - loss: 0.1054 - acc: 0.9550 - val_loss: 0.2080 - val_acc: 0.9449
Epoch 17/30
40/40 [==============================] - 1s 30ms/step - loss: 0.0974 - acc: 0.9672 - val_loss: 0.1911 - val_acc: 0.9567
Epoch 18/30
40/40 [==============================] - 1s 30ms/step - loss: 0.1014 - acc: 0.9621 - val_loss: 0.2059 - val_acc: 0.9449
Epoch 19/30
40/40 [==============================] - 1s 30ms/step - loss: 0.0984 - acc: 0.9652 - val_loss: 0.2043 - val_acc: 0.9449
Epoch 20/30
40/40 [==============================] - 1s 30ms/step - loss: 0.0903 - acc: 0.9713 - val_loss: 0.1854 - val_acc: 0.9567
Epoch 21/30
40/40 [==============================] - 1s 30ms/step - loss: 0.0750 - acc: 0.9744 - val_loss: 0.1905 - val_acc: 0.9528
Epoch 22/30
40/40 [==============================] - 1s 31ms/step - loss: 0.0828 - acc: 0.9693 - val_loss: 0.1837 - val_acc: 0.9646
Epoch 23/30
40/40 [==============================] - 1s 31ms/step - loss: 0.0865 - acc: 0.9662 - val_loss: 0.1927 - val_acc: 0.9528
Epoch 24/30
40/40 [==============================] - 1s 31ms/step - loss: 0.0744 - acc: 0.9734 - val_loss: 0.1790 - val_acc: 0.9528
Epoch 25/30
40/40 [==============================] - 1s 32ms/step - loss: 0.0767 - acc: 0.9703 - val_loss: 0.1916 - val_acc: 0.9528
Epoch 26/30
40/40 [==============================] - 1s 31ms/step - loss: 0.0747 - acc: 0.9683 - val_loss: 0.1841 - val_acc: 0.9567
Epoch 27/30
40/40 [==============================] - 1s 32ms/step - loss: 0.0612 - acc: 0.9795 - val_loss: 0.1958 - val_acc: 0.9567
Epoch 28/30
40/40 [==============================] - 1s 32ms/step - loss: 0.0568 - acc: 0.9826 - val_loss: 0.1933 - val_acc: 0.9567
Epoch 29/30
40/40 [==============================] - 1s 32ms/step - loss: 0.0572 - acc: 0.9806 - val_loss: 0.2020 - val_acc: 0.9528
Epoch 30/30
40/40 [==============================] - 1s 31ms/step - loss: 0.0685 - acc: 0.9734 - val_loss: 0.1841 - val_acc: 0.9606
======================== <05> 训练模型 ========================
8/8 [==============================] - 18s 2s/step - loss: 0.1684 - acc: 0.9429
======================== <06> 查看模型性能 ========================
======================== <07> 查看混淆矩阵 ========================
Exporing the model to ./models
2023-04-25 10:05:42.297148: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
2023-04-25 10:05:44.063711: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2023-04-25 10:05:44.063889: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-04-25 10:05:44.097131: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
  function_optimizer: Graph size after: 225 nodes (167), 367 edges (297), time = 10.345ms.
  function_optimizer: Graph size after: 225 nodes (0), 367 edges (0), time = 6.801ms.
Optimization results for grappler item: __inference_Assert_AssertGuard_true_663_104
  function_optimizer: function_optimizer did nothing. time = 0.011ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_Assert_AssertGuard_false_96_180
  function_optimizer: function_optimizer did nothing. time = 0.003ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_cond_true_217_208
  function_optimizer: Graph size after: 9 nodes (0), 10 edges (0), time = 0.318ms.
  function_optimizer: Graph size after: 9 nodes (0), 10 edges (0), time = 0.244ms.
Optimization results for grappler item: __inference_Assert_AssertGuard_true_298_41
  function_optimizer: function_optimizer did nothing. time = 0.003ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_Assert_AssertGuard_true_95_116
  function_optimizer: function_optimizer did nothing. time = 0.002ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_Assert_1_AssertGuard_true_114_192
  function_optimizer: function_optimizer did nothing. time = 0.002ms.
  function_optimizer: function_optimizer did nothing. time = 0ms.
Optimization results for grappler item: __inference_Assert_1_AssertGuard_false_115_86
  function_optimizer: function_optimizer did nothing. time = 0.003ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_Assert_AssertGuard_false_664_174
  function_optimizer: function_optimizer did nothing. time = 0.003ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_cond_false_218_162
  function_optimizer: Graph size after: 15 nodes (0), 14 edges (0), time = 0.305ms.
  function_optimizer: Graph size after: 15 nodes (0), 14 edges (0), time = 0.306ms.
Optimization results for grappler item: __inference_cond_Assert_AssertGuard_false_225_73
  function_optimizer: function_optimizer did nothing. time = 0.003ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_cond_Assert_AssertGuard_true_224_198
  function_optimizer: function_optimizer did nothing. time = 0.003ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_Assert_AssertGuard_true_197_186
  function_optimizer: function_optimizer did nothing. time = 0.002ms.
  function_optimizer: function_optimizer did nothing. time = 0ms.
Optimization results for grappler item: __inference_Assert_AssertGuard_false_299_47
  function_optimizer: function_optimizer did nothing. time = 0.004ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_cond_Assert_AssertGuard_true_249_110
  function_optimizer: function_optimizer did nothing. time = 0.003ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_cond_Assert_AssertGuard_false_250_146
  function_optimizer: function_optimizer did nothing. time = 0.002ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.
Optimization results for grappler item: __inference_Assert_AssertGuard_false_198_18
  function_optimizer: function_optimizer did nothing. time = 0.003ms.
  function_optimizer: function_optimizer did nothing. time = 0.001ms.

2023-04-25 10:05:44.386163: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
2023-04-25 10:05:44.386204: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
======================== <08> 导出模型 ========================
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
---prediction---
Class: off
Score: 0.999983549118042
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/off/3a789a0d_nohash_1.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 1/80 ========================
---prediction---
Class: go
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/go/b1426003_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 2/80 ========================
---prediction---
Class: on
Score: 0.9999178647994995
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/on/19e246ad_nohash_1.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 3/80 ========================
---prediction---
Class: right
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/right/16db1582_nohash_2.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 4/80 ========================
---prediction---
Class: off
Score: 0.9801470637321472
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/off/340c8b10_nohash_2.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 5/80 ========================
---prediction---
Class: stop
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/stop/617de221_nohash_4.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 8/80 ========================
---prediction---
Class: go
Score: 0.9999397993087769
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/go/0ff728b5_nohash_3.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 10/80 ========================
---prediction---
Class: go
Score: 0.9999967813491821
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/go/cd68e997_nohash_1.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 11/80 ========================
---prediction---
Class: on
Score: 0.9999998807907104
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/on/76424fa5_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 13/80 ========================
---prediction---
Class: on
Score: 0.9999972581863403
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/on/c0e8f5a1_nohash_1.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 14/80 ========================
---prediction---
Class: right
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/right/587f3271_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 16/80 ========================
---prediction---
Class: go
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/go/611d2b50_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 17/80 ========================
---prediction---
Class: right
Score: 0.9999830722808838
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/right/80c17118_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 19/80 ========================
---prediction---
Class: go
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/go/63996b7c_nohash_2.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 20/80 ========================
---prediction---
Class: right
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/right/ce49cb60_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 21/80 ========================
---prediction---
Class: stop
Score: 0.9974374771118164
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/stop/3852fca2_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 22/80 ========================
---prediction---
Class: off
Score: 0.999983549118042
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/off/3a789a0d_nohash_1.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 25/80 ========================
---prediction---
Class: stop
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/stop/2bdbe5f7_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 26/80 ========================
---prediction---
Class: on
Score: 0.9999998807907104
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/on/76424fa5_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 27/80 ========================
---prediction---
Class: down
Score: 0.9997484087944031
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/down/8dd24423_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 28/80 ========================
---prediction---
Class: down
Score: 0.9996399879455566
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/down/b9f46737_nohash_1.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 30/80 ========================
---prediction---
Class: down
Score: 0.9972061514854431
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/down/71904de3_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 31/80 ========================
---prediction---
Class: stop
Score: 0.9999105930328369
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/stop/62581901_nohash_1.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 32/80 ========================
---prediction---
Class: right
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/right/06a79a03_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 33/80 ========================
---prediction---
Class: up
Score: 0.995992124080658
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/up/b9515bf3_nohash_2.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 38/80 ========================
---prediction---
Class: off
Score: 0.9801470637321472
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/off/340c8b10_nohash_2.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 41/80 ========================
---prediction---
Class: off
Score: 0.9801470637321472
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/off/340c8b10_nohash_2.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 44/80 ========================
---prediction---
Class: down
Score: 1.0
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/down/aff582a1_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 51/80 ========================
---prediction---
Class: stop
Score: 0.9999997615814209
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/stop/72198b96_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 56/80 ========================
---prediction---
Class: left
Score: 0.9999996423721313
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/left/ce49cb60_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 69/80 ========================
---prediction---
Class: left
Score: 0.9999992847442627
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/left/3411cf4b_nohash_1.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 71/80 ========================
---prediction---
Class: left
Score: 0.9999994039535522
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/left/8281a2a8_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 74/80 ========================
---prediction---
Class: down
Score: 0.9999951124191284
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/down/ece1a95a_nohash_1.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 78/80 ========================
---prediction---
Class: up
Score: 0.9913163185119629
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/up/e5e54cee_nohash_0.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 79/80 ========================
---prediction---
Class: up
Score: 0.9919982552528381
----truth----
File: /media/ihdd/01_rtos/zephyr-diverse/zephyr/resources/tflite/04_speech_recognization/dataset-test/up/8e05039f_nohash_3.wav
Sample rate: 16000
Sample length: 16000
======================== <09> 使用 TF Lite 模型运行推断 80/80 ========================
