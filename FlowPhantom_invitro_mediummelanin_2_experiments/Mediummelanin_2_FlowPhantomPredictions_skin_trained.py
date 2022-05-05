import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt

def spectrum_normalisation(spectrum):

    # Applies z-score scaling to the initial pressure spectrum
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    norm = (spectrum - mean)/std
    return norm


def timestep_preprocessing(timebatch, wavelengths):
    processed = []
    indices = [int((i-700)/5) for i in wavelengths]
    for spectrum in timebatch:
        spectrum_with_zeroes = [0 for i in range(41)]
        norm = spectrum_normalisation(list(spectrum))
        count = 0
        for index in indices:
            spectrum_with_zeroes[index] = norm[count]
            count += 1
        processed.append(torch.tensor(spectrum_with_zeroes))
    return torch.stack(processed)

# INSERT LOADING MODEL CODE (model trained on melanin set)
saved_model = tf.keras.models.load_model('../TrainedModels/filteredoriginal_11.h5', compile = False)

measured_wavelengths = [700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900]
rounded_wavelengths = [700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900]
timesteps = 284
so2_predictions = []

so2_stddevs = []
for i in range(timesteps):
    print(i)
    filename = '../Datasets/FlowPhantom_invitro_mediummelanin_2/Timestep' + str(i) + '.pt'
    data = torch.load(filename)
    data = timestep_preprocessing(data, rounded_wavelengths)
    data = torch.reshape(data, (len(data), len(data[0]), 1))
    data = data.numpy()
    data = tf.convert_to_tensor(data)

    # MODEL SO2 ESTIMATION ON EACH PIXEL'S SPECTRUM, THEN AVERAGE OVER THE PIXELS TO GET THE TIMESTEP SO2 ESTIMATE
    predictions = tf.convert_to_tensor(saved_model.predict(data))
    '''
    if i ==0:
        np.save('Time0_SASD_predictions.npy', np.array(predictions))
    '''
    if i == 283:
        np.save('Time283_SASD_predictions.npy', np.array(predictions))
    timestep_stddev = tf.math.reduce_std(predictions)
    timestep_so2 = tf.math.reduce_mean(predictions)
    so2_stddevs.append(timestep_stddev)
    so2_predictions.append(timestep_so2)

#np.save('Results/LSTM_stddev.npy', np.array(so2_stddevs))
'''
np.save('Results/LSTM_skintraining_mediummelanin_2_invitro_flowpredictions.npy', np.array(so2_predictions))

lsd_so2_predictions = np.load('Results/LSD_old_phantomtraining_invitro_flowpredictions.npy')
lu_so2_predictions = np.load('Results/LU_invitro_flowpredictions.npy')

'''
plt.plot(so2_predictions, label = 'LSTM', linewidth = 0.5)

'''
plt.plot(lu_so2_predictions, label = 'LU', linewidth = 0.5)
plt.plot(lsd_so2_predictions,label = 'LSD',linewidth = 0.5)
'''
plt.legend(loc='upper right')
plt.xlabel('Time')
plt.ylabel('sO2')
ax = plt.gca()
ax.set_ylim([0,1])
plt.show()