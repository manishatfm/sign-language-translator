import tensorflow as tf

# List all available devices (should include GPU if available)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("✅ GPU detected:", physical_devices)
else:
    print("❌ No GPU detected, using CPU.")
