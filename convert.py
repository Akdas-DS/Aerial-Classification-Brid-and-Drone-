import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("final_transfer_model.keras", compile=False)

# Create the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Make the model smaller + faster (optional but recommended)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

# Convert model
tflite_model = converter.convert()

# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✔ Conversion complete! Saved as model.tflite")
