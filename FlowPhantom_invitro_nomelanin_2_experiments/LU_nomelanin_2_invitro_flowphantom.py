# Script performs an approximate linear unmixing method (using inversion) to estimate sO2
import torch
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.optimize import minimize
import itertools

#######################################################################################################################

# Altering tabulated data to match the wavelengths used
HbO2_absorption_coefficients = [290, 348, 446, 586, 710, 816, 916, 1022, 1092, 1154, 1198]
Hb_absorption_coefficients = [1794.28, 1325.88, 1115.88, 1548.52, 1075.44, 761.72, 693.76, 692.36,694.32, 726.44, 761.84]

absorption_list = [x for x in itertools.chain.from_iterable(itertools.zip_longest(HbO2_absorption_coefficients,Hb_absorption_coefficients)) if x]

########################################################################################################################
# Now implementing LU

# Function used to minimise the absolute difference between the simulated initial pressures and the spectra predicted from LU
def predictedp0_vs_simulated(coefficients, absorption_coefficients_and_simulated_spectra):
    aHgO2 = coefficients[0]
    aHg = coefficients[1]

    number_of_mua = int(2*len(absorption_coefficients_and_simulated_spectra)/3)
    absorption_coefficients = [absorption_coefficients_and_simulated_spectra[i] for i in range(number_of_mua)]
    simulated_spectra = [absorption_coefficients_and_simulated_spectra[i] for i in range(number_of_mua,len(absorption_coefficients_and_simulated_spectra))]

    predicted_spectra = [aHgO2 * absorption_coefficients[2*i] + aHg * absorption_coefficients[2*i + 1] for i in range(int(0.5 * len(absorption_coefficients)))]
    return np.linalg.norm(list(map(operator.sub, predicted_spectra, simulated_spectra)))

timesteps = 232
averaged_predictions = []
stddevs = []
for i in range(timesteps):
    print(i)
    filename = '../Datasets/FlowPhantom_invitro_nomelanin_2/Timestep' + str(i) + '.pt'
    f = torch.load(filename)
    timestep_predictions = []
    initial_guess = [0.1, 0.1]
    bnds = ((0, None), (0, None))

    for j in range(len(f)):
        initial_pressures = list(f[j])
        # Tuple of the absorption coefficients and the initial pressure spectrum which are arguments in the function
        func_input = absorption_list + initial_pressures
        # Calling minimisation function
        min = minimize(predictedp0_vs_simulated, initial_guess, args=func_input,bounds=bnds, options= {'eps':0.1})
        timestep_predictions.append(min.x[0]/(min.x[0]+min.x[1]))
        '''print(min.x[0], min.x[1])'''
    if i ==0:
        np.save('Time0_LU_predictions.npy', np.array(timestep_predictions))
    if i ==200:
        np.save('Time200_LU_predictions.npy', np.array(timestep_predictions))
    averaged_predictions.append(np.mean(timestep_predictions))
    stddevs.append(np.std(timestep_predictions))

averaged_predictions = np.array(averaged_predictions)
stddevs = np.array(stddevs)
np.save('Results/LU_stddev.npy',stddevs)
np.save('Results/LU_invitro_flowpredictions.npy', averaged_predictions)
plt.plot(averaged_predictions, linewidth = 0.5)
plt.show()