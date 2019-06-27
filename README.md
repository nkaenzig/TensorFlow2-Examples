# Tensorflow 2.0

## tf.data.Dataset

### Pipelines
Recommended way to set up efficient data pipelines for Tensorflow.

Example 1: Create pipeline for jpeg images
```python
def parse_function(file_path, label):
    image_string = tf.read_file(file_path)

    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [64, 64])
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(parse_function)
dataset = dataset.cache()
# dataset = dataset.cache(filename='./cache.tf-data') # if the data doesn't fit in memory, use a cache file
dataset = dataset.shuffle(len(image_paths))
dataset = dataset.repeat(10)
# dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size, count))
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
```

* without dataset.cache(), each image file would be read again in each epoch (after each repeat()) (is fine for local training on CPU but may not be sufficient for GPU training, and is totally inappropriate for any sort of distributed training.)
    
* from_tensor_slices accepts different types of input arguments 
    * numpy array or tensor
    * placeholder (useful when we want to dynamically change the data inside the Dataset)
    * generator (useful when we have an array of different elements length)
    * list of filepaths  (useful when big dataset that doesn’t fit in memory)

### Iterators
* In TF <2.0 Iterators such as make_one_shot_iterator(), were used to give us the ability to iterate through the dataset and retrieve the real values of the data. (inside session.run())
* In TF 2.0 Datasets are iterables  and eager mode -> creating iterators has become obsolete
    * They were removed from tf.data API!

## tf.keras

### Sequential model API (symbolic)
The simplest model API - preferred way to build simple models such as MLP, CNNs, RNNs, ...

* You can create a Sequential model by passing a list of layer instances to the constructor.
```python     
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

* You can also simply add layers via the .add() method:
```python     
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

### Model class API (functional API) (symbolic)
The Keras functional API is the way to go for defining complex models, such as multi-output models, directed acyclic graphs, or models with shared layers.
[Keras Doc](https://keras.io/getting-started/functional-api-guide/)
[TF Doc](https://www.tensorflow.org/guide/keras#build_advanced_models)

* All models are callable, just like layers
* Multi-input and multi-output models
* Supervision via multiple loss functions (at different layers in the model)
* Shared layers
* Models with non-sequential data flows (e.g. residual connections)

```python     
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

main_input = Input(shape=(100,), dtype='int32', name='main_input')

x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)

auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```

* E.g.: Inception module:
```python  
from keras.layers import Conv2D, MaxPooling2D, Input

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
```

### Model subclassing API (imperative)
[Subclassing API Doc](https://www.tensorflow.org/guide/keras#model_subclassing)
[Keras Models Doc](https://keras.io/models/about-keras-models/)

The Model subclassing API can provide you with greater flexbility for implementing complex models (implementing your own forward pass in the call method), but it comes at a cost: 
* It is more verbose, more complex, and has more opportunities for user errors. 
* No model.inputs, model.outputs, model.to_yaml(), model.to_json(), model.get_config(), model.save()

"Subclassing tf.keras objects (Layer, Model, Metric...), together with eager execution & the GradientTape, bridges the gap from the lowest-level APIs to the highest-level features like Sequential and fit(). Full flexibility + productivity."

* If possible, use the functional API, which is more user-friendly.
* Model subclassing is particularly useful when eager execution is enabled since the forward pass can be written imperatively.

The call methods of a tf.keras.Model subclass can be decorated with tf.function in order to apply graph execution optimizations on it:
```python     
class MyModel(tf.keras.Model):
  def __init__(self, num_classes=10, keep_probability=0.2):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(num_classes, activation='softmax')
    self.keep_probability = keep_probability

  @tf.function
  def call(self, x, training=true):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x
    if training:
        return tf.nn.dropout(x, self.keep_probability)
    else:
        return x

model = MyModel()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.fit(data, labels, batch_size=32, epochs=5)
```

### Training
Models defined in either the Sequential, Functional, or Subclassing style can be trained in two ways. You can use either a built-in training routine and loss function (see the first example, where we use model.fit and model.compile), or if you need the added complexity of a custom training loop (for example, if you’d like to write your own gradient clipping code) or loss function, you can do so easily as follows:

```python  
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
  
@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

                                              
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for images, labels in train_ds:
      train_step(images, labels)
    
    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels)

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_accuracy.result()*100,
                           test_loss.result(),
                           test_accuracy.result()*100))
```

## tf.function()
[tf.function doc](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function)
- Creates a callable TensorFlow graph from a Python function.
- A tf.function you define is just like a core TensorFlow operation: You can execute it eagerly; you can use it in a graph; it has gradients; and so on.
