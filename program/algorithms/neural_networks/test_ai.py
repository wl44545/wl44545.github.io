import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import inspect
from tqdm import tqdm
import os


path, dirs, files = next(os.walk("/usr/lib"))
file_count = len(files)


# Set batch size for training and validation
batch_size = 16

# List all available models
model_dictionary = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}


# Number of training examples and labels
num_train = len(list(train))
num_validation = len(list(validation))
num_classes = len(metadata.features['label'].names)
num_iterations = int(num_train/batch_size)


# Print important info
print(f'Num train images: {num_train} \
        \nNum validation images: {num_validation} \
        \nNum classes: {num_classes} \
        \nNum iterations per epoch: {num_iterations}')

def normalize_img(image, label, img_size):
    # Resize image to the desired img_size and normalize it
    # One hot encode the label
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.
    label = tf.one_hot(label, depth=num_classes)
    return image, label

def preprocess_data(train, validation, batch_size, img_size):
    # Apply the normalize_img function on all train and validation data and create batches
    # train_processed = train.map(lambda image, label: normalize_img(image, label, img_size))
    # train_processed = train_processed.batch(batch_size).repeat()
    #
    # validation_processed = validation.map(lambda image, label: normalize_img(image, label, img_size))
    # validation_processed = validation_processed.batch(batch_size)

    train_processed = tf.keras.preprocessing.image_dataset_from_directory(
        "train",
        labels="inferred",
        label_mode="categorical",  # categorical, binary
        # class_names=['0', '1', '2', '3', ...]
        batch_size=batch_size,
        image_size=(img_size, img_size),  # reshape if not in this size
        shuffle=True,
        seed=123,
        # validation_split=0.1,
        # subset="training",
    )

    validation_processed = tf.keras.preprocessing.image_dataset_from_directory(
        "val",
        labels="inferred",
        label_mode="categorical",  # categorical, binary
        # class_names=['0', '1', '2', '3', ...]
        batch_size=batch_size,
        image_size=(img_size, img_size),  # reshape if not in this size
        shuffle=True,
        seed=123,
        # validation_split=0.1,
        # subset="validation",
    )

    def augment(x, y):
        image = tf.image.random_brightness(x, max_delta=0.05)
        return image, y

    train_processed = train_processed.map(augment)

    return train_processed, validation_processed

# Run preprocessing
train_processed_224, validation_processed_224 = preprocess_data(train, validation, batch_size, img_size=224)
train_processed_331, validation_processed_331 = preprocess_data(train, validation, batch_size, img_size=331)



tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit", histogram_freq=1)

# Loop over each model available in Keras
model_benchmarks = {'model_name': [], 'num_model_params': [], 'validation_accuracy': []}
for model_name, model in tqdm(model_dictionary.items()):
    print(model_name)
    # Special handling for "NASNetLarge" since it requires input images with size (331,331)
    if 'NASNetLarge' in model_name:
        input_shape=(331,331,3)
        train_processed = train_processed_331
        validation_processed = validation_processed_331
    else:
        input_shape=(224,224,3)
        train_processed = train_processed_224
        validation_processed = validation_processed_224

    # load the pre-trained model with global average pooling as the last layer and freeze the model weights
    pre_trained_model = model(include_top=False, pooling='avg', input_shape=input_shape)
    pre_trained_model.trainable = False

    # custom modifications on top of pre-trained model
    clf_model = tf.keras.models.Sequential()
    clf_model.add(pre_trained_model)
    clf_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    clf_model.compile(loss='categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.TruePositives()])
    history = clf_model.fit(train_processed, epochs=1, validation_data=validation_processed,
                            steps_per_epoch=num_iterations,callbacks=[tensorboard_callback])

    # Calculate all relevant metrics
    model_benchmarks['model_name'].append(model_name)
    model_benchmarks['num_model_params'].append(pre_trained_model.count_params())
    model_benchmarks['validation_accuracy'].append(history.history['val_accuracy'][-1])

    break

# Convert Results to DataFrame for easy viewing
benchmark_df = pd.DataFrame(model_benchmarks)
benchmark_df.sort_values('num_model_params', inplace=True) # sort in ascending order of num_model_params column
benchmark_df.to_csv('benchmark_df.csv', index=False) # write results to csv file
benchmark_df

# Loop over each row and plot the num_model_params vs validation_accuracy
markers=[".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",4,5,6,7,8,9,10,11]
plt.figure()
for row in benchmark_df.itertuples():
    plt.scatter(row.num_model_params, row.validation_accuracy, label=row.model_name, marker=markers[row.Index], s=150, linewidths=2)
plt.xscale('log')
plt.xlabel('Number of Parameters in Model')
plt.ylabel('Validation Accuracy after 3 Epochs')
plt.title('Accuracy vs Model Size')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left'); # Move legend out of the plot
plt.show()
