import tensorflow as tf
import numpy as np
import torch
import random
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from ViolinPlot import *

def spectrum_normalisation(spectrum):
    # Applies z-score scaling to the initial pressure spectrum
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    norm = (spectrum - mean)/std
    return norm
'''
def spectrum_normalisation(spectrum):
    # Applies min-max scaling to the initial pressure spectrum
    norm = minmax_scale(spectrum, feature_range=(0.01,0.99))
    return norm
'''

def median_error_fraction(y_true, y_pred):
    error = abs((y_true - y_pred) / y_true)
    return tfp.stats.percentile(error, 50.0, interpolation='midpoint')


def testset_error_fraction(y_true, y_pred):
    # Function for finding the median and IQR for so2 error
    error = abs((y_true - y_pred) / y_true)
    percentile_25 = tfp.stats.percentile(error, 25.0, interpolation='midpoint')
    median = tfp.stats.percentile(error, 50.0, interpolation='midpoint')
    percentile_75 = tfp.stats.percentile(error, 75.0, interpolation='midpoint')
    return [percentile_25, median, percentile_75]


def spectrum_processing(spectrum, allowed_datapoints):

    # Returns a normalised initial pressure spectrum with some of the values zeroed out

    num_non_zero_datapoints = random.choice(allowed_datapoints)
    a = np.zeros(len(spectrum))
    a[:num_non_zero_datapoints] = 1
    np.random.shuffle(a)

    incomplete_spectrum = list(np.multiply(a, np.array(spectrum)))
    non_zero_indices = np.nonzero(incomplete_spectrum)
    non_zero_values = list(filter(None,incomplete_spectrum))
    normalised_non_zero = spectrum_normalisation(non_zero_values)

    i = 0
    for index in non_zero_indices[0]:
        incomplete_spectrum[index] = normalised_non_zero[i]
        i+=1

    normalised_incomplete_spectrum = np.array(incomplete_spectrum)

    return normalised_incomplete_spectrum

def batch_spectrum_processing(batch, allowed_datapoints):
    processed = []

    for spectrum in batch:

        processed.append(spectrum_processing(spectrum, allowed_datapoints))
    return torch.tensor(np.array(processed))

# Initialise list to hold the IQR of model errors (on test set)
IQRs = []

# Setting the random seeds for reproducible results
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

# Importing spectra and oxygenations (pytorch tensors)
train_spectra_original = torch.load('../Datasets/FlowPhantom_insilico_complicated/training_spectra.pt')
validation_spectra_original = torch.load('../Datasets/FlowPhantom_insilico_complicated/validation_spectra.pt')
test_spectra_original = torch.load('../Datasets/FlowPhantom_insilico_complicated/test_spectra.pt')
train_oxygenations = torch.load('../Datasets/FlowPhantom_insilico_complicated/training_oxygenations.pt')
validation_oxygenations = torch.load('../Datasets/FlowPhantom_insilico_complicated/validation_oxygenations.pt')
test_oxygenations = torch.load('../Datasets/FlowPhantom_insilico_complicated/test_oxygenations.pt')
print(train_spectra_original[98])
training_samples = len(train_spectra_original)
validation_samples = len(validation_spectra_original)
test_samples = len(test_spectra_original)

# Zeroing out some of the spectrum data (randomly) and normalises
allowed_datapoints = [10]

train_spectra = batch_spectrum_processing(train_spectra_original, allowed_datapoints)
validation_spectra = batch_spectrum_processing(validation_spectra_original, allowed_datapoints)
test_spectra = batch_spectrum_processing(test_spectra_original, allowed_datapoints)

# Reshaping initial pressure spectra to fit LSTM input size
train_spectra = torch.reshape(train_spectra, (len(train_spectra), len(train_spectra[0]), 1))
validation_spectra = torch.reshape(validation_spectra, (len(validation_spectra), len(validation_spectra[0]), 1))
test_spectra = torch.reshape(test_spectra, (len(test_spectra), len(test_spectra[0]), 1))

train_oxygenations = torch.reshape(train_oxygenations,(len(train_oxygenations),1))
validation_oxygenations = torch.reshape(validation_oxygenations,(len(validation_oxygenations),1))
test_oxygenations = torch.tensor(np.float32(test_oxygenations))
test_oxygenations = torch.reshape(test_oxygenations,(len(test_oxygenations),1))

# Converting pytorch tensors to tensorflow tensors
train_spectra = train_spectra.numpy()
train_spectra = tf.convert_to_tensor(train_spectra)
validation_spectra = validation_spectra.numpy()
validation_spectra = tf.convert_to_tensor(validation_spectra)
test_spectra = test_spectra.numpy()
test_spectra = tf.convert_to_tensor(test_spectra)
train_oxygenations = tf.convert_to_tensor(train_oxygenations)
validation_oxygenations = tf.convert_to_tensor(validation_oxygenations)
test_oxygenations = tf.convert_to_tensor(test_oxygenations)


print(train_spectra[98])
print(train_spectra.get_shape())
print(train_oxygenations)
print(train_oxygenations.get_shape())

# Creating the tensorflow Dataset objects
ds_train = tf.data.Dataset.from_tensor_slices((train_spectra, train_oxygenations))
ds_validation = tf.data.Dataset.from_tensor_slices((validation_spectra, validation_oxygenations))
ds_test = tf.data.Dataset.from_tensor_slices((test_spectra, test_oxygenations))

# Appropriately batching the Datasets
batch_size = 2048

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(buffer_size=training_samples)
ds_train = ds_train.batch(batch_size, drop_remainder=True)

ds_validation = ds_validation.batch(batch_size, drop_remainder=True)
ds_validation = ds_validation.cache()

ds_test = ds_test.cache()
ds_test = ds_test.shuffle(buffer_size=test_samples)
ds_test = ds_test.batch(batch_size, drop_remainder=True)

# Defining the model
model3 = tf.keras.models.Sequential()
model3.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(41, 1)))
model3.add(tf.keras.layers.LSTM(100, return_sequences=True, activation='relu'))
model3.add(tf.keras.layers.Flatten())
model3.add(tf.keras.layers.LeakyReLU())
model3.add(tf.keras.layers.Dense(1000))
model3.add(tf.keras.layers.LeakyReLU())
model3.add(tf.keras.layers.Dense(1000))
model3.add(tf.keras.layers.LeakyReLU())
model3.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))

# Model parameters
additional_metrics = [tf.keras.metrics.MeanAbsolutePercentageError()]
loss_function = tf.keras.losses.MeanAbsoluteError()
number_of_epochs = 100
optimizer = Adam(learning_rate=0.001)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001, verbose=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=22, verbose=1)
mc = tf.keras.callbacks.ModelCheckpoint('Flow_insilico_complicated_10.h5', monitor = 'val_median_error_fraction', mode = 'min', verbose = 1, save_best_only = True)
# Check device available
print(tf.config.list_physical_devices('GPU'))

# Compiling model
model3.compile(optimizer=optimizer, loss=loss_function, metrics=[additional_metrics, median_error_fraction])

# Printing model structure
print(model3.summary())

'''
for i, l in enumerate(model1.layers):
    print(f'layer {i}: {l}')
    print(f'has input mask: {l.input_mask}')
    print(f'has output mask: {l.output_mask}')
'''
# Training the model
history3 = model3.fit(ds_train, epochs=number_of_epochs, validation_data=ds_validation, callbacks=[reduce_lr, es,mc])
# print(history.history) gives the history of the losses over the epochs

# Loading saved model and evaluating on test set
saved_model = tf.keras.models.load_model('Flow_insilico_complicated_10.h5', compile = False)
predictions = tf.convert_to_tensor(saved_model.predict(test_spectra))
print(predictions)
print(test_oxygenations)
print(np.shape(predictions))

IQRs.append(testset_error_fraction(test_oxygenations, predictions))
print(IQRs)
print(r2_score(test_oxygenations,predictions))

reshaped_predictions = np.array(tf.reshape(predictions,[28848])) *100

reshaped_gts =np.array(tf.reshape(test_oxygenations,[28848]))*100


a = create_violin_scatter_plot('lstm_phantom_10.png', reshaped_predictions,reshaped_gts)
