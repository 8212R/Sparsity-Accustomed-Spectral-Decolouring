import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from ViolinPlot import *
'''
def spectrum_normalisation(spectrum):
# Applies (-1,1) min-max scaling to the initial pressure spectrum
    norm = minmax_scale(spectrum,feature_range=(-1,1))
    return norm
'''


def spectrum_normalisation(spectrum):
    # Applies z-score scaling to the initial pressure spectrum
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    norm = (spectrum - mean)/std
    return norm


def spectrum_processing(spectrum,allowed_indices):
# Takes in the full 41-long spectrum, and returns the normalised incomplete spectrum
    temp = []
    for i in range(len(spectrum)):
        if i in allowed_indices:
            temp.append(spectrum[i])
    temp = spectrum_normalisation(temp)
    return temp

def batch_processing(batch,allowed_indices):
# Returns incomplete + normalised initial pressure spectra from the original dataset
    processed = []
    for spectrum in batch:
        processed.append(spectrum_processing(spectrum, allowed_indices))

    return torch.tensor(np.array(processed))

def testset_error_fraction(y_true, y_pred):
# Function for finding the median and IQR for so2 error
    error = abs((y_true - y_pred) / y_true)
    q = torch.tensor([0.25,0.50,0.75])
    IQR = torch.quantile(error,q)
    return IQR

# Importing the full initial pressure spectra
train_spectra_original = torch.load('../Datasets/NoSkin_filtered/filtered_training_spectra.pt')
validation_spectra_original =  torch.load('../Datasets/NoSkin_filtered/filtered_validation_spectra.pt')
test_spectra_original =  torch.load('../Datasets/NoSkin_filtered/filtered_test_spectra.pt')

print(train_spectra_original[53])

# Importing the ground truth oxygenations, and reshaping so that each spectrum has a label
train_oxygenations =  torch.load('../Datasets/NoSkin_filtered/filtered_training_oxygenations.pt')
validation_oxygenations = torch.load('../Datasets/NoSkin_filtered/filtered_validation_oxygenations.pt')
test_oxygenations = torch.load('../Datasets/NoSkin_filtered/filtered_test_oxygenations.pt')
train_oxygenations= torch.reshape(train_oxygenations,(len(train_oxygenations),1))
validation_oxygenations = torch.reshape(validation_oxygenations,(len(validation_oxygenations),1))
test_oxygenations=torch.reshape(test_oxygenations,(len(test_oxygenations),1))
test_oxygenations = np.float32(test_oxygenations)
test_oxygenations = torch.tensor(test_oxygenations)
# Removing some wavelength data depending on the number of datapoints the network will be trained on
N_datapoints = 2

indices_41 = [i for i in range(41)]
indices_40 = [i for i in range(41) if i != 20]
indices_39 = [i for i in range(41) if i not in [13,26]]
indices_38 = [i for i in range(41) if i not in [10, 20, 30]]
indices_37 = [i for i in range(41) if i not in [8, 16, 24, 32]]
indices_36 = [i for i in range(41) if i not in [7,14,21,28,35]]
indices_35 = [i for i in range(41) if i not in [6,11,17,23,29,34]]
indices_34 = [i for i in range(41) if i not in [5,10,15,20,25,30,35]]
indices_33 = [i for i in range(41) if i not in [5,9,14,18,23,27,31,36]]
indices_32 = [i for i in range(41) if i not in [4,8,12,16,20,24,28,32,36]]
indices_31 = [i for i in range(41) if i not in [3,7,11,15,19,23,26,30,34,39]]
indices_30 = [i for i in range(41) if i not in [4,7,11,15,18,22,25,29,33,36,40]]
indices_29 = [i for i in range(41) if i not in [4,7,10,13,16,19,22,25,28,31,34,37]]
indices_28 = [i for i in range(41) if i not in [3,6,9,12,15,18,21,24,27,30,33,36,39]]
indices_27 = [i for i in range(41) if i not in [3,5,8,11,13,16,19,22,25,28,31,34,37,40]]
indices_26 = [i for i in range(41) if i not in [3,5,8,10,13,16,19,21,24,27,39,32,35,38,40]]
indices_25 = [i for i in range(41) if i not in [3,5,8,10,13,15,18,20,22,25,27,30,32,35,38,40]]
indices_24 = [i for i in range(41) if i not in [3,5,7,9,12,14,17,19,22,24,26,29,32,34,36,38,40]]
indices_23 = [i for i in range(41) if i not in [1,3,5,8,10,13,15,18,20,22,25,27,30,32,34,36,38,40]]
indices_22 = [i for i in range(41) if i not in [1,3,5,8,10,12,14,17,19,21,23,25,27,29,31,33,36,38,40]]
indices_21 = [i for i in range(41) if i not in [1,3,5,7,9,11,13,15,17,19,21,24,26,28,30,32,34,36,38,40]]
indices_20 = [i for i in range(41) if i not in [2,4,5,7,9,11,13,15,16,18,20,22,24,25,27,29,31,33,35,36,38]]
indices_19 = [i for i in range(41) if i not in [0,2,4,5,7,9,11,13,15,16,18,20,22,24,25,27,29,31,33,35,36,38]]
indices_18 = [i for i in range(41) if i not in [0,2,4,5,7,9,11,13,15,16,18,20,22,24,25,27,29,31,33,35,36,38,40]]
indices_17 = [i for i in range(41) if i not in [0,2,4,5,6,7,9,11,13,15,16,18,20,22,24,25,27,29,31,33,35,36,38,40]]
indices_16 = [i for i in range(41) if i not in [0,2,4,5,6,7,9,11,13,15,16,18,19,20,22,24,25,27,29,31,33,35,36,38,40]]
indices_15 = [i for i in range(41) if i not in [1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33,34,36,37,39]]
indices_14 = [i for i in range(41) if i not in [1,3,4,6,7,9,10,11,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33,34,36,37,39]]
indices_13 = [i for i in range(41) if i not in [1,3,4,6,7,9,10,11,12,13,15,16,18,19,21,22,23,24,25,27,28,30,31,33,34,36,37,39]]
indices_12 = [i for i in range(41) if i not in [1,3,4,6,7,9,10,11,12,13,15,16,18,19,21,22,24,25,26,27,28,30,31,32,33,34,36,37,39]]
indices_11 = [i for i in range(41) if i not in [0,1,3,4,6,7,9,10,11,12,13,15,16,18,19,21,22,24,25,27,28,29,30,31,33,34,35,36,37,39]]
indices_10 = [i for i in range(41) if i not in [1,3,4,5,6,8,9,10,11,13,14,15,16,18,19,20,21,23,24,25,26,28,29,30,31,33,34,35,36,38,39]]
indices_9 = [4,8,12,16,20,24,28,32,36]
indices_8 = [5,9,14,18,23,27,31,36]
indices_7 = [5,10,15,20,25,30,35]
indices_6 = [6,11,17,23,29,34]
indices_5 = [0,10,20,30,40]
indices_4 = [8,16,24,32]
indices_3 = [0,20,40]
indices_2 = [13, 27]
if N_datapoints == 41:
    allowed_datapoints = indices_41
if N_datapoints == 40:
    allowed_datapoints = indices_40
if N_datapoints == 39:
    allowed_datapoints = indices_39
if N_datapoints == 38:
    allowed_datapoints = indices_38
if N_datapoints == 37:
    allowed_datapoints = indices_37
if N_datapoints == 36:
    allowed_datapoints = indices_36
if N_datapoints == 35:
    allowed_datapoints = indices_35
if N_datapoints == 34:
    allowed_datapoints = indices_34
if N_datapoints == 33:
    allowed_datapoints = indices_33
if N_datapoints == 32:
    allowed_datapoints = indices_32
if N_datapoints == 31:
    allowed_datapoints = indices_31
if N_datapoints == 30:
    allowed_datapoints = indices_30
if N_datapoints == 29:
    allowed_datapoints = indices_29
if N_datapoints == 28:
    allowed_datapoints = indices_28
if N_datapoints == 27:
    allowed_datapoints = indices_27
if N_datapoints == 26:
    allowed_datapoints = indices_26
if N_datapoints == 25:
    allowed_datapoints = indices_25
if N_datapoints == 24:
    allowed_datapoints = indices_24
if N_datapoints == 23:
    allowed_datapoints = indices_23
if N_datapoints == 22:
    allowed_datapoints = indices_22
if N_datapoints == 21:
    allowed_datapoints = indices_21
if N_datapoints == 20:
    allowed_datapoints = indices_20
if N_datapoints == 19:
    allowed_datapoints = indices_19
if N_datapoints == 18:
    allowed_datapoints = indices_18
if N_datapoints == 17:
    allowed_datapoints = indices_17
if N_datapoints == 16:
    allowed_datapoints = indices_16
if N_datapoints == 15:
    allowed_datapoints = indices_15
if N_datapoints == 14:
    allowed_datapoints = indices_14
if N_datapoints == 13:
    allowed_datapoints = indices_13
if N_datapoints == 12:
    allowed_datapoints = indices_12
if N_datapoints == 11:
    allowed_datapoints = indices_11
if N_datapoints == 10:
    allowed_datapoints = indices_10
if N_datapoints == 9:
    allowed_datapoints = indices_9
if N_datapoints == 8:
    allowed_datapoints = indices_8
if N_datapoints == 7:
    allowed_datapoints = indices_7
if N_datapoints == 6:
    allowed_datapoints = indices_6
if N_datapoints == 5:
    allowed_datapoints = indices_5
if N_datapoints == 4:
    allowed_datapoints = indices_4
if N_datapoints == 3:
    allowed_datapoints = indices_3
if N_datapoints == 2:
    allowed_datapoints = indices_2

train_spectra = batch_processing(train_spectra_original,allowed_datapoints)
validation_spectra = batch_processing(validation_spectra_original,allowed_datapoints)
test_spectra = batch_processing(test_spectra_original,allowed_datapoints)

print(train_spectra[53])

# Initialising dataloaders
train_ds = TensorDataset(train_spectra,train_oxygenations)
validation_ds = TensorDataset(validation_spectra,validation_oxygenations)
test_ds = TensorDataset(test_spectra,test_oxygenations)

batch_size = 1024

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
valid_loader = DataLoader(validation_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size,shuffle=True)
print('Data imported and loaded')

# Defining LSD network
class LSD(nn.Module):

    def __init__(self):
        super().__init__()

        self.LSD = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=N_datapoints,out_features=N_datapoints*2),
            nn.LeakyReLU(),
            nn.Linear(in_features=N_datapoints*2, out_features=N_datapoints*2),
            nn.LeakyReLU(),
            nn.Linear(in_features=N_datapoints*2, out_features=N_datapoints*2),
            nn.LeakyReLU(),
            nn.Linear(in_features=N_datapoints*2, out_features=1)
        )

    def forward(self, x):
        x = self.LSD(x)
        return x

### Define the loss function
loss_fn = torch.nn.L1Loss()

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the network
network = LSD()

params_to_optimize = [
    {'params': network.parameters()}
]

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move the network to the selected device
network.to(device)

### Training function
def train_epoch_den(network, device, dataloader, loss_fn, optimizer):
    network.train()
    train_loss = []
    # Iterate the dataloader
    for batch in dataloader:
        spectrum_batch = batch[0]
        labels = batch[1]
        # Move tensor to the proper device
        spectrum_batch = spectrum_batch.to(device)
        labels = labels.to(device)
        output_data = network(spectrum_batch.float())
        # Evaluate loss
        loss = loss_fn(output_data.float(), labels.float())
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

### Testing function
def test_epoch_den(network, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    network.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        i = 0
        for batch in dataloader:
            spectrum_batch = batch[0]
            labels = batch[1]
            # Move tensor to the proper device
            spectrum_batch = spectrum_batch.to(device)
            labels = labels.to(device)
            output_data = network(spectrum_batch.float())

            if i == 0 and flag == True:
                print(labels[0])
                print('label printed')
                print(output_data[0])
                print('output printed')
            i += 1

            # Append the network output and the original to the lists
            conc_out.append(output_data.cpu())
            conc_label.append(labels.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        conc_error_fractions = abs((conc_out - conc_label) / conc_label)

        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data, torch.median(conc_error_fractions)

### Training cycle
num_epochs = 100
history_da = {'train_loss': [], 'val_loss': []}
flag = True

for epoch in range(num_epochs):
    print('EPOCH %d/%d' % (epoch + 1, num_epochs))

    if epoch % 2 == 1:
        lr = 0.01 * 0.9 ** ((epoch - 1) / 2)
    else:
        lr = 0.01 * 0.9 ** (epoch / 2)

    optim = torch.optim.Adam(params_to_optimize, lr=lr)

    ### Training (use the training function)
    train_loss = train_epoch_den(
        network=network,
        device=device,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optim)
    print('Training done')

    ### Validation  (use the testing function)
    val_loss, median_error = test_epoch_den(
        network=network,
        device=device,
        dataloader=valid_loader,
        loss_fn=loss_fn)

    # Print Validationloss
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss, val_loss))
    print('Median error fraction: ' + str(float(median_error)))


# Evaluate performance on test set
network.eval()
test_spectra = test_spectra.to(device)
predictions = network(test_spectra.float())
print(predictions)
print(type(predictions))
print(test_oxygenations)
print(type(test_oxygenations))
IQR = testset_error_fraction(test_oxygenations,predictions)
print(IQR)
'''
predictions = torch.reshape(predictions,[12013])
test_oxygenations = torch.reshape(test_oxygenations,[12013])
reshaped_predictions = predictions.detach().numpy() *100

reshaped_gts =test_oxygenations.detach().numpy()*100


a = create_violin_scatter_plot('lsd_noskin_10.png', reshaped_predictions,reshaped_gts)
'''