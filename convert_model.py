import tensorflow as tf

model = tf.keras.models.load_model("final_transfer_model.h5")

# IMPORTANT — Save in Keras v3 format (new format)
model.save("final_transfer_model_fixed.keras", save_format="keras")

print("Model converted successfully! Saved as final_transfer_model_fixed.keras")
