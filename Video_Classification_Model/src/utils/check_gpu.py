import tensorflow as tf

# TensorFlow version
print("TensorFlow version:", tf.__version__)

# CUDA version
cuda_version = tf.sysconfig.get_build_info().get("cuda_version", "Unknown")
print("CUDA version:", cuda_version)

# cuDNN version
cudnn_version = tf.sysconfig.get_build_info().get("cudnn_version", "Unknown")
print("cuDNN version:", cudnn_version)

# Verify that TensorFlow can see the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
tf.print(physical_devices)

# Simple matrix multiplication to test GPU usage
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    c = tf.matmul(a, b)
    print(c)

