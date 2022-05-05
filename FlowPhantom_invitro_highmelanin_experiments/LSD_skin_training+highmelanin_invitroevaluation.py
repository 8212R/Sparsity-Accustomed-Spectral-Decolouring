import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
'''
def spectrum_normalisation(spectrum):
# Applies (-1,1) min-max scaling to the initial pressure spectrum
    norm = minmax_scale(spectrum,feature_range=(-1,1))
    return norm
'''
def timestep_preprocessing(timebatch):
    processed = []
    for spectrum in timebatch:
        processed.append(torch.tensor(spectrum_normalisation(list(spectrum))))
    return torch.stack(processed)

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
train_spectra_original = torch.load('../Datasets/Skin_filtered/train_spectra_original_filtered.pt')
validation_spectra_original = torch.load('../Datasets/Skin_filtered/validation_spectra_original_filtered.pt')
test_spectra_original = torch.load('../Datasets/Skin_filtered/test_spectra_original_filtered.pt')

print(train_spectra_original[53])

# Importing the ground truth oxygenations, and reshaping so that each spectrum has a label
train_oxygenations = torch.load('../Datasets/Skin_filtered/train_oxygenations_original_filtered.pt')
validation_oxygenations = torch.load('../Datasets/Skin_filtered/validation_oxygenations_original_filtered.pt')
test_oxygenations = torch.load('../Datasets/Skin_filtered/test_oxygenations_original_filtered.pt')
train_oxygenations= torch.reshape(train_oxygenations,(len(train_oxygenations),1))
validation_oxygenations = torch.reshape(validation_oxygenations,(len(validation_oxygenations),1))
test_oxygenations=torch.reshape(test_oxygenations,(len(test_oxygenations),1))

# Removing some wavelength data depending on the number of datapoints the network will be trained on
N_datapoints = 11


indices_11 = [0,4,8,12,16,20,24,28,32,36,40]
if N_datapoints == 11:
    allowed_datapoints = indices_11

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

# Flow phantom in vitro set
timesteps = 87
so2_predictions = []
for i in range(timesteps):
    print(i)
    filename = '../Datasets/FlowPhantom_invitro_highmelanin/Timestep' + str(i) + '.pt'
    data = torch.load(filename)
    data = timestep_preprocessing(data)
    # MODEL SO2 ESTIMATION ON EACH PIXEL'S SPECTRUM, THEN AVERAGE OVER THE PIXELS TO GET THE TIMESTEP SO2 ESTIMATE
    invitro_predictions = network(data.float())
    timestep_so2 = torch.mean(invitro_predictions)
    so2_predictions.append(timestep_so2.cpu().detach().numpy())

np.save('Results/LSD_skintraining_highmelanin_invitro_flowpredictions.npy', np.array(so2_predictions))
plt.plot(so2_predictions)
plt.xlabel('Time')
plt.ylabel('sO2')
plt.show()
