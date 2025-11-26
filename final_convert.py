import keras

OLD_MODEL = "final_transfer_model.keras"   # or .keras if that is your old file
NEW_MODEL = "final_transfer_model_keras3.keras"

print("Loading old model...")
model = keras.models.load_model(OLD_MODEL, compile=False)

print("Saving model in new Keras 3 format...")
model.save(NEW_MODEL)

print("Done! Saved as:", NEW_MODEL)
