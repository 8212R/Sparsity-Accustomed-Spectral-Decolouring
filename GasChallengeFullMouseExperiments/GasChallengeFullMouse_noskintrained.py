import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import itertools
import h5py
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('rb',['b','w','r'], N=256)


def spectrum_normalisation(spectrum):

    # Applies z-score scaling to the initial pressure spectrum
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    norm = (spectrum - mean)/std
    return norm


def timestep_preprocessing(timebatch,wavelengths_used):

    indices = [int((i - 700)/5) for i in wavelengths_used]
    processed = []

    for spectrum in timebatch:

        spectrum_with_zeroes = []
        spectrum_normalised = spectrum_normalisation(list(spectrum))
        count = 0

        for i in range(41):
            if i in indices:
                spectrum_with_zeroes.append(spectrum_normalised[count])
                count += 1
            else:
                spectrum_with_zeroes.append(0)

        processed.append(torch.tensor(spectrum_with_zeroes))

    return torch.stack(processed)


# Data for the PA reconstruction
f = h5py.File('I:/research/seblab/data/group_folders/Janek/kevindata/Scan_108.hdf5','r')
reconstructed_data = f['recons']['Backprojection Preclinical']['0']

# INSERT LOADING MODEL CODE (model trained on 10 random wavelengths on the generic tissue set)
saved_model = tf.keras.models.load_model('../TrainedModels/NoSkinFiltered_10.h5', compile = False)

# Parameters for gas challenge study
measured_wavelengths = [700, 730, 750, 760, 770, 800, 820, 840, 850, 880]
timesteps = 93

# Plotting the segmentation lines for the tumour and reference region
s1 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines1.npy')
s2 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines2.npy')
s3 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines3.npy')
s4 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines4.npy')
s5 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines5.npy')
s6 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines6.npy')
s7 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines7.npy')
s8 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines8.npy')
s9 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines9.npy')
s10 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/Outlines10.npy')
t1 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/tumouroutlines1a.npy')
t2 = np.load('../Datasets/GasChallengeFullMouse/Segmentation/tumouroutlines1b.npy')

# Collecting the coordinates of the points within the segmented regions
chosen_pixel_coords = np.load('../Datasets/GasChallengeFullMouse/Segmentation/pixel_coords.npy')
chosen_pixel_coords = list(chosen_pixel_coords)
chosen_pixel_coords = [list(entry) for entry in chosen_pixel_coords]
print(len(chosen_pixel_coords))
tumour_coords = np.load('../Datasets/GasChallengeFullMouse/Segmentation/tumourcoords2.npy')
tumour_coords = list(tumour_coords)
tumour_coords = [list(entry) for entry in tumour_coords]

tumour_indices = []
for i in range(len(tumour_coords)):
    tumour_indices.append(chosen_pixel_coords.index(tumour_coords[i]))
'''
# For each timestep, the model estimates so2 for each pixel:

so2_timeseries_bypixel = [[] for i in range(len(chosen_pixel_coords))]

for i in range(timesteps):
    print(i)
    filename = '../Datasets/GasChallengeFullMouse/Timestep' + str(i) + '.pt'
    data = torch.load(filename)
    data = timestep_preprocessing(data, measured_wavelengths)
    data = torch.reshape(data, (len(data), len(data[0]), 1))
    data = data.numpy()
    data = tf.convert_to_tensor(data)
    # MODEL SO2 ESTIMATION ON EACH PIXEL'S SPECTRUM
    predictions = tf.convert_to_tensor(saved_model.predict(data))
    for j in range(len(predictions)):
        so2_timeseries_bypixel[j].append(np.float64(predictions[j]))

#np.save('../GasChallengeFullMouse_results/SASD/so2_timeseries_bypixel.npy', np.array(so2_timeseries_bypixel))

# Plotting the so2_maps
for i in range(timesteps):
    background_image_data = reconstructed_data[i][5] # Background greyscale PA image at 800nm
    plt.imshow(background_image_data,cmap='gray',origin = 'lower')

    so2_data_fortimestep = [pixel[i] for pixel in so2_timeseries_bypixel]

    so2_map = [[np.NaN for i in range(250)] for j in range(250)]

    count = 0
    for entry in chosen_pixel_coords:
        x_coord = entry[0]
        y_coord = entry[1]
        so2_map[y_coord][x_coord] = so2_data_fortimestep[count] *100
        count += 1

    plt.imshow(so2_map,interpolation = 'nearest', vmin = 0, vmax = 100, origin = 'lower')
    clb = plt.colorbar()
    clb.ax.set_title('sO$_2$ [%]')
    plt.plot(*zip(*s1), color='r')
    plt.plot(*zip(*s2), color='r')
    plt.plot(*zip(*s3), color='r')
    plt.plot(*zip(*s4), color='r')
    plt.plot(*zip(*s5), color='r')
    plt.plot(*zip(*s6), color='r')
    plt.plot(*zip(*s7), color='r')
    plt.plot(*zip(*s8), color='r')
    plt.plot(*zip(*s9), color='r')
    plt.plot(*zip(*s10), color='r')
    file = '../GasChallengeFullMouse_results/SASD/Absoluteso2maps/GasChallengeFullMouse_Timestep' + str(i)
    #plt.savefig(file)
    plt.show()
'''

so2_timeseries_bypixel = np.load('../GasChallengeFullMouse_results/SASD/so2_timeseries_bypixel.npy')

# Now implementing a 'baseline so2 comparison'
N_baseline_timesteps = 46

# Time-averaged absolute so2 plot after baseline:

absolute_time_average_afterbaseline = [np.mean(pixeldata[N_baseline_timesteps:]) for pixeldata in so2_timeseries_bypixel]

background_image_data = reconstructed_data[0][5] # Background greyscale PA image at 800nm
plt.imshow(background_image_data,cmap='gray',origin = 'lower')
so2_map = [[np.NaN for i in range(250)] for j in range(250)]

count = 0
for entry in chosen_pixel_coords:
    x_coord = entry[0]
    y_coord = entry[1]
    so2_map[y_coord][x_coord] = absolute_time_average_afterbaseline[count] *100
    count += 1

plt.imshow(so2_map,interpolation = 'nearest', vmin = 0, vmax = 100, origin = 'lower')
clb = plt.colorbar()
clb.ax.set_title('sO$_2$ [%]')
plt.plot(*zip(*s1), color='k')
plt.plot(*zip(*s2), color='k')
plt.plot(*zip(*s3), color='k')
plt.plot(*zip(*s4), color='k')
plt.plot(*zip(*s5), color='k')
plt.plot(*zip(*s6), color='k')
plt.plot(*zip(*s7), color='k')
plt.plot(*zip(*s8), color='k')
plt.plot(*zip(*s9), color='k')
plt.plot(*zip(*s10), color='k')
plt.plot(*zip(*t1), color = 'k')
plt.plot(*zip(*t2), color = 'k')
file = '../GasChallengeFullMouse_results/SASD/time_averaged_absolute.png'
#plt.savefig(file)
plt.show()

pixel_baselines = []
for i in range(len(so2_timeseries_bypixel)):
    baseline = np.mean(so2_timeseries_bypixel[i][0:N_baseline_timesteps])
    pixel_baselines.append(baseline)

delta_so2_timeseries_bypixel = []
for i in range(len(so2_timeseries_bypixel)):
    delta_so2_series = list(np.array(so2_timeseries_bypixel[i]) - pixel_baselines[i])
    delta_so2_series = delta_so2_series[N_baseline_timesteps:]
    delta_so2_timeseries_bypixel.append(delta_so2_series)

#np.save('../GasChallengeFullMouse_results/SASD/Baseline46_delta_so2_timeseries_bypixel.npy', np.array(delta_so2_timeseries_bypixel))
mindeltas = []
maxdeltas = []
# Plotting the delta_so2_maps
'''
for i in range(timesteps-N_baseline_timesteps):
    background_image_data = reconstructed_data[i+N_baseline_timesteps][5] # Background greyscale PA image at 800nm


    delta_so2_data_fortimestep = [pixel[i] for pixel in delta_so2_timeseries_bypixel]

    delta_so2_map = [[np.NaN for i in range(250)] for j in range(250)]
    mindeltas.append(min(delta_so2_data_fortimestep))
    maxdeltas.append(max(delta_so2_data_fortimestep))
    count = 0
    for entry in chosen_pixel_coords:
        x_coord = entry[0]
        y_coord = entry[1]
        delta_so2_map[y_coord][x_coord] = delta_so2_data_fortimestep[count]*100
        count += 1
    fig,axes = plt.subplots()
    plt.imshow(background_image_data, cmap='gray', origin='lower')
    plt.imshow(delta_so2_map,interpolation = 'nearest', vmin = -70, vmax = 70, origin = 'lower', cmap=cmap)
    clb = plt.colorbar()
    clb.ax.set_title('\u0394sO$_2$ [%]')
    plt.plot(*zip(*s1), color='r')
    plt.plot(*zip(*s2), color='r')
    plt.plot(*zip(*s3), color='r')
    plt.plot(*zip(*s4), color='r')
    plt.plot(*zip(*s5), color='r')
    plt.plot(*zip(*s6), color='r')
    plt.plot(*zip(*s7), color='r')
    plt.plot(*zip(*s8), color='r')
    plt.plot(*zip(*s9), color='r')
    plt.plot(*zip(*s10), color='r')
    file = '../GasChallengeFullMouse_results/SASD/Deltaso2maps/GasChallengeFullMouse_deltaso2_baseline46_Timestep' + str(i+N_baseline_timesteps)
    #plt.savefig(file)
    plt.show()

print(max(maxdeltas))
print(min(mindeltas))
'''

# Creating a heat map and histogram of the timeaveraged absolute change in so2 for each pixel
time_averaged_delta_so2_bypixel =  [np.mean(entry) for entry in delta_so2_timeseries_bypixel]
time_averaged_absolute_delta_so2_bypixel = [np.mean(abs(np.array(entry))) for entry in delta_so2_timeseries_bypixel]


time_averaged_delta_so2_tumourpixels = []
time_averaged_delta_so2_nontumourpixels = []
time_averaged_absolute_delta_so2_tumourpixels = []
time_averaged_absolute_delta_so2_nontumourpixels = []

for i in range(len(time_averaged_absolute_delta_so2_bypixel)):
    if i in tumour_indices:
        time_averaged_absolute_delta_so2_tumourpixels.append(time_averaged_absolute_delta_so2_bypixel[i])
        time_averaged_delta_so2_tumourpixels.append(time_averaged_delta_so2_bypixel[i])
    else:
        time_averaged_absolute_delta_so2_nontumourpixels.append(time_averaged_absolute_delta_so2_bypixel[i])
        time_averaged_delta_so2_nontumourpixels.append(time_averaged_delta_so2_bypixel[i])

background_image_data = reconstructed_data[0][5] # Background greyscale PA image at 800nm
delta_so2_map = [[np.NaN for i in range(250)] for j in range(250)]

count = 0
for entry in chosen_pixel_coords:
    x_coord = entry[0]
    y_coord = entry[1]
    delta_so2_map[y_coord][x_coord] =time_averaged_delta_so2_bypixel[count] *100
    count += 1

fig, axes = plt.subplots()
plt.imshow(background_image_data, cmap='gray', origin='lower')
plt.imshow(delta_so2_map, interpolation='nearest', vmin=-70, vmax=70, origin='lower', cmap=cmap)
clb = plt.colorbar()
clb.ax.set_title('\u0394sO$_2$ [%]')
plt.plot(*zip(*s1), color='k')
plt.plot(*zip(*s2), color='k')
plt.plot(*zip(*s3), color='k')
plt.plot(*zip(*s4), color='k')
plt.plot(*zip(*s5), color='k')
plt.plot(*zip(*s6), color='k')
plt.plot(*zip(*s7), color='k')
plt.plot(*zip(*s8), color='k')
plt.plot(*zip(*s9), color='k')
plt.plot(*zip(*s10), color='k')
plt.plot(*zip(*t1), color = 'k')
plt.plot(*zip(*t2), color = 'k')
file = '../GasChallengeFullMouse_results/SASD/time_averaged_delta.png'
#plt.savefig(file)
plt.show()


print(np.mean(time_averaged_absolute_delta_so2_tumourpixels))
print(np.mean(time_averaged_absolute_delta_so2_nontumourpixels))
plt.hist(np.array(time_averaged_delta_so2_tumourpixels)*100, bins = 100, range = [-50,50], facecolor = 'gray', align = 'mid')
plt.xlabel('Time-averaged \u0394sO$_2$ [%] ')
plt.ylabel('Frequency')
#plt.savefig('../GasChallengeFullMouse_results/SASD/Baseline46_TumourHistogram0.png')
plt.show()

plt.hist(np.array(time_averaged_delta_so2_nontumourpixels) * 100, bins=100, range=[-50, 50], facecolor='gray', align='mid')
plt.xlabel('Time-averaged \u0394sO$_2$ [%] ')
plt.ylabel('Frequency')
#plt.savefig('../GasChallengeFullMouse_results/SASD/Baseline46_NonTumourHistogram0.png')
plt.show()

np.save('pvalue_tumour.npy',np.array(time_averaged_absolute_delta_so2_tumourpixels)*100 )
np.save('pvalue_nontumour.npy',np.array(time_averaged_absolute_delta_so2_nontumourpixels)*100 )
plt.hist(np.array(time_averaged_absolute_delta_so2_tumourpixels)*100, bins = 100, range = [0,50], facecolor = 'gray', align = 'mid')
plt.xlabel('Time-averaged |\u0394sO$_2$| [%] ')
plt.ylabel('Frequency')
#plt.savefig('../GasChallengeFullMouse_results/SASD/Baseline46_TumourHistogram.png')
plt.show()
plt.hist(np.array(time_averaged_absolute_delta_so2_nontumourpixels)*100, bins = 100, range = [0,50], facecolor = 'gray', align = 'mid')
plt.xlabel('Time-averaged |\u0394sO$_2$| [%] ')
plt.ylabel('Frequency')
#plt.savefig('../GasChallengeFullMouse_results/SASD/Baseline46_NonTumourHistogram.png')
plt.show()