from scripts.global_imports import*
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


images,masks = create_dataset()

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2)

augmented_images, augmented_masks = data_augmentation(X_train, y_train)

X_to_validation, X_to_test, y_to_validation, y_to_test = train_test_split(X_test, y_test, test_size=0.5)

X_to_validation = np.array(X_to_validation)
X_to_test = np.array(X_to_test)

y_to_test = np.array(y_to_test)
y_to_validation = np.array(y_to_validation)




initial_learning_rate = 0.0001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


model.compile(optimizer=optimizer,
              loss=combined_loss(dice_weight=0.5, focal_weight=0.5),
              metrics=['accuracy',jaccard_index,dice_coefficient])

sample_image, sample_mask = X_to_test[3], y_to_test[3]


checkpoint = ModelCheckpoint(
    'models/segment_model.keras',          
    monitor='val_dice_coefficient',         
    verbose=1,                   
    save_best_only=True,         
    mode='max',                 
    save_weights_only=False,     
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

model.fit(augmented_images, augmented_masks, epochs=80, validation_data=(X_to_validation, y_to_validation), batch_size=16, callbacks=[checkpoint, early_stopping])