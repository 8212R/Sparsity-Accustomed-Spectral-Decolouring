import numpy as np
import h5py
import matplotlib.pyplot as plt

chosen_pixels = [[155,169]]
chosen_pixels += [[156,i] for i in range(166,173)]
chosen_pixels += [[157,i] for i in range(164, 175)]
chosen_pixels += [[158,i] for i in range(163, 176)]
chosen_pixels += [[159,i] for i in range(163, 176)]
chosen_pixels += [[160,i] for i in range(162, 177)]
chosen_pixels += [[161,i] for i in range(162, 177)]
chosen_pixels += [[162,i] for i in range(162, 177)]
chosen_pixels += [[163,i] for i in range(161, 178)]
chosen_pixels += [[164,i] for i in range(162, 177)]
chosen_pixels += [[165,i] for i in range(162, 177)]
chosen_pixels += [[166,i] for i in range(162, 177)]
chosen_pixels += [[167,i] for i in range(163, 176)]
chosen_pixels += [[168,i] for i in range(163, 176)]
chosen_pixels += [[169,i] for i in range(164, 175)]
chosen_pixels += [[170,i] for i in range(166,173)]
chosen_pixels += [[171,169]]
for entry in chosen_pixels:
    entry[0] -= 13
    entry[1] += 11

plot_pixel_coords = [[entry[0]-134,entry[1]-164] for entry in chosen_pixels]

# PA image at t = 283
f = h5py.File('I:/research/seblab/data/group_folders/Tom/Melanin_Flow Phantom/Processed_data/Scan_7.hdf5')
data = f['recons']['OpenCL Backprojection']['0']

plotdata = np.reshape(data[283][7],(333,333))
clipped_plotdata = plotdata.copy()[165:195,135:165]
fig,ax = plt.subplots()
plt.imshow(clipped_plotdata, origin = 'lower')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
circle = plt.Circle((16,16),8.0, fill = False, edgecolor= 'r', linewidth=8.0)
clb = plt.colorbar()
ax.add_patch(circle)
file = 'T=283_PAimage_MELANIN.png'
plt.savefig(file)
plt.show()

# SASD image at t = 283
a = np.load('Time283_SASD_predictions.npy')*100
print(np.mean(a))
print(max(a)- np.mean(a))
print(np.mean(a)- min(a))
SASD_t0_plotdata = [[np.NaN for i in range(30)] for j in range(30)]

for i in range(len(plot_pixel_coords)):
    x_coord = plot_pixel_coords[i][0]
    y_coord = plot_pixel_coords[i][1]
    SASD_t0_plotdata[y_coord][x_coord] = np.float32(a[i])


fig2,ax2 = plt.subplots()
plt.imshow(np.float32(SASD_t0_plotdata),interpolation = 'nearest', vmin = np.mean(a)-10.0, vmax = np.mean(a) + 7.5, origin = 'lower')
clb = plt.colorbar()
clb.ax.set_title('sO$_2$ [%]')
circle = plt.Circle((16,16),8.0, fill = False, edgecolor= 'r', linewidth=8.0)
ax2.add_patch(circle)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
file = 'T=283_SASD_MELANIN.png'
plt.savefig(file)
plt.show()


# LSD image at t = 283
b = np.load('Time283_LSD_predictions.npy')*100
print(max(b)-np.mean(b))
print(np.mean(b)-min(b))
LSD_t0_plotdata = [[np.NaN for i in range(30)] for j in range(30)]

for i in range(len(plot_pixel_coords)):
    x_coord = plot_pixel_coords[i][0]
    y_coord = plot_pixel_coords[i][1]
    LSD_t0_plotdata[y_coord][x_coord] = np.float32(b[i])


fig3,ax3 = plt.subplots()
plt.imshow(np.float32(LSD_t0_plotdata),interpolation = 'nearest', vmin = np.mean(b)-10, vmax = np.mean(b)+7.5, origin = 'lower')
clb = plt.colorbar()
clb.ax.set_title('sO$_2$ [%]')
circle = plt.Circle((16,16),8.0, fill = False, edgecolor= 'r', linewidth=8.0)
ax3.add_patch(circle)
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)
file = 'T=283_LSD_MELANIN.png'
plt.savefig(file)
plt.show()


# LU image at t = 283
c = np.load('Time283_LU_predictions.npy')*100
print(max(c)-np.mean(c))
print(np.mean(c)-min(c))
LU_t0_plotdata = [[np.NaN for i in range(30)] for j in range(30)]

for i in range(len(plot_pixel_coords)):
    x_coord = plot_pixel_coords[i][0]
    y_coord = plot_pixel_coords[i][1]
    LU_t0_plotdata[y_coord][x_coord] = np.float32(c[i])

fig4,ax4 = plt.subplots()
plt.imshow(np.float32(LU_t0_plotdata),interpolation = 'nearest', vmin = np.mean(c)-10, vmax = np.mean(c)+7.5, origin = 'lower')
clb = plt.colorbar()
clb.ax.set_title('sO$_2$ [%]')
circle = plt.Circle((16,16),8.0, fill = False, edgecolor= 'r', linewidth=8.0)
ax4.add_patch(circle)
ax4.xaxis.set_visible(False)
ax4.yaxis.set_visible(False)
file = 'T=283_LU_MELANIN.png'
plt.savefig(file)
plt.show()
