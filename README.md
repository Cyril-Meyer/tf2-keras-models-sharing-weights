# tf2-keras-models-sharing-weights

Based on "Training a neural network on MNIST with Keras" : https://www.tensorflow.org/datasets/keras_example


```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
```


```python
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
```


```python
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
```


```python
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
```


```python
model_input = tf.keras.layers.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(model_input)
x = tf.keras.layers.Dense(128, activation='relu')(x)

model_output_1 = tf.keras.layers.Dense(10)(x)
model_output_2 = tf.keras.layers.Dense(10)(x)

model_1 = tf.keras.Model(model_input, model_output_1, name="model_1")
model_1.summary()

model_2 = tf.keras.Model(model_input, model_output_2, name="model_2")
model_2.summary()

model_1.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model_2.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28)]          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               100480    
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________
    Model: "model_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28)]          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               100480    
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________



```python
print(np.array_equal(np.array(model_1.layers[2].get_weights()[0]).flatten(), np.array(model_2.layers[2].get_weights()[0]).flatten()))
print(np.array_equal(np.array(model_1.layers[2].get_weights()[1]).flatten(), np.array(model_2.layers[2].get_weights()[1]).flatten()))
```

    True
    True



```python
print(np.array_equal(np.array(model_1.layers[3].get_weights()[0]).flatten(), np.array(model_2.layers[3].get_weights()[0]).flatten()))
print(np.array_equal(np.array(model_1.layers[3].get_weights()[1]).flatten(), np.array(model_2.layers[3].get_weights()[1]).flatten()))
```

    False
    True



```python
print(model_2.evaluate(ds_train))
print(model_2.evaluate(ds_train))
```

    469/469 [==============================] - 1s 2ms/step - loss: 2.3934 - sparse_categorical_accuracy: 0.1142
    [2.393404960632324, 0.11423332989215851]
    469/469 [==============================] - 1s 2ms/step - loss: 2.3934 - sparse_categorical_accuracy: 0.1142
    [2.3934051990509033, 0.11423332989215851]



```python
model_1.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
```

    Epoch 1/6
    469/469 [==============================] - 2s 3ms/step - loss: 0.3603 - sparse_categorical_accuracy: 0.8998 - val_loss: 0.1894 - val_sparse_categorical_accuracy: 0.9480
    Epoch 2/6
    469/469 [==============================] - 1s 3ms/step - loss: 0.1610 - sparse_categorical_accuracy: 0.9544 - val_loss: 0.1358 - val_sparse_categorical_accuracy: 0.9606
    Epoch 3/6
    469/469 [==============================] - 1s 3ms/step - loss: 0.1170 - sparse_categorical_accuracy: 0.9663 - val_loss: 0.1076 - val_sparse_categorical_accuracy: 0.9692
    Epoch 4/6
    469/469 [==============================] - 1s 3ms/step - loss: 0.0920 - sparse_categorical_accuracy: 0.9734 - val_loss: 0.0937 - val_sparse_categorical_accuracy: 0.9722
    Epoch 5/6
    469/469 [==============================] - 1s 3ms/step - loss: 0.0740 - sparse_categorical_accuracy: 0.9789 - val_loss: 0.0859 - val_sparse_categorical_accuracy: 0.9742
    Epoch 6/6
    469/469 [==============================] - 1s 3ms/step - loss: 0.0610 - sparse_categorical_accuracy: 0.9827 - val_loss: 0.0791 - val_sparse_categorical_accuracy: 0.9764





    <tensorflow.python.keras.callbacks.History at 0x7f13a004c3c8>




```python
print(model_2.evaluate(ds_train))
print(model_2.evaluate(ds_train))
```

    469/469 [==============================] - 1s 2ms/step - loss: 4.0878 - sparse_categorical_accuracy: 0.0431
    [4.08781099319458, 0.04309999942779541]
    469/469 [==============================] - 1s 2ms/step - loss: 4.0878 - sparse_categorical_accuracy: 0.0431
    [4.087810516357422, 0.04309999942779541]



```python
print(np.array_equal(np.array(model_1.layers[2].get_weights()[0]), np.array(model_2.layers[2].get_weights()[0])))
print(np.array_equal(np.array(model_1.layers[2].get_weights()[1]), np.array(model_2.layers[2].get_weights()[1])))
```

    True
    True



```python
print(np.array_equal(np.array(model_1.layers[3].get_weights()[0]).flatten(), np.array(model_2.layers[3].get_weights()[0]).flatten()))
print(np.array_equal(np.array(model_1.layers[3].get_weights()[1]).flatten(), np.array(model_2.layers[3].get_weights()[1]).flatten()))
```

    False
    False

