import tensorflow as tf

# 1. Load your existing Keras model
model = tf.keras.models.load_model("final_transfer_model.keras", compile=False)

# 2. Save it again in the new Keras format (Keras 3–friendly)
model.save("final2_transfer_model_fixed.keras", save_format="keras")

print("✅ Converted and saved as final_transfer_model_fixed.keras")
