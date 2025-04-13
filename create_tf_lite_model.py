import tensorflow as tf
from scripts.loss_functions import*

model = tf.keras.models.load_model('models/segment_model.keras', custom_objects={"loss":combined_loss,"jaccard_index":jaccard_index})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('models/segment_models.tflite', 'wb') as f:
    f.write(tflite_model)

print("--------------------------TFLite model saved--------------------------")