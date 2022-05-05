from scipy.io import loadmat
import glob
from os.path import join
import pandas as pd
import numpy as np
from scipy import signal


def severinghaus(po2):
    # Apply the severinghaus equation to convert the po2 into so2
    return 100 / (1 + 23400 / (po2 ** 3 + 150 * po2))


def load_matlab_table(filename, key):
    simplified_data = loadmat(filename, simplify_cells=True)
    # For the data types:
    detailed_data = loadmat(filename, simplify_cells=False)
    column_names = simplified_data[key][0]
    data = simplified_data[key][1:]
    types = [d.dtype for d in detailed_data[key][1]]
    df = pd.DataFrame(data=data, columns=column_names)
    for data_type, column in zip(types, column_names):
        df[column] = df[column].astype(data_type)
    return df


def load_po2(mat_files):

    dfs = []
    for mat_file in mat_files:
        df = load_matlab_table(mat_file, "pO2data")
        dfs.append(df)
    df = dfs[0].append(dfs[1:])
    df["Time"] = pd.to_datetime(df["Time"], format='%H:%M:%S:%f')
    df = df.sort_values("Time", ignore_index=True)
    df["Time"] -= df["Time"][0]
    df["so2 (Pre)"] = severinghaus(df["mmHg (Pre)"])
    df["so2 (Post)"] = severinghaus(df["mmHg (Post)"])
    return df


class FlowDataAnalyser:
    def __init__(self, pa_data, po2_data_folder, view_wavelength=800,
                 recon=None, trim_start=0, roi_name="unnamed_", roi_number="0", probe_time_subtract=0):
        pa_data.set_default_recon(recon)
        self.pa_data = pa_data
        self.scan_name = self.pa_data.get_scan_name()
        self.po2_data = load_po2(self.scan_name, po2_data_folder)
        wl_number = np.argmin(np.abs(self.pa_data.get_wavelengths() - view_wavelength))
        self.pa_image = self.pa_data.get_scan_reconstructions()

        # Extract the sO2 data:
        so2 = self.pa_data.get_scan_so2()
        roi = self.pa_data.get_rois()[roi_name, roi_number]
        roi_mask, so2_data = roi.to_mask_slice(so2)
        so2_trace = np.mean(so2_data.raw_data.T[roi_mask.T].T, axis=(1, 2))
        times = self.pa_data.get_timestamps()[:so2_trace.shape[0], 0]
        times -= times[0]

        thb = self.pa_data.get_scan_thb()
        roi_mask, thb_data = roi.to_mask_slice(thb)
        thb_trace = np.mean(thb_data.raw_data.T[roi_mask.T].T, axis=(1, 2))

        self.pa_data_table = pd.DataFrame(data={"Time": times, "so2_pa": so2_trace,
                                                "thb": thb_trace})
        self.pa_data_table["Time"] = pd.to_timedelta(self.pa_data_table["Time"], unit="s")

        roi_mask, recon_data = roi.to_mask_slice(self.pa_image)
        wavelengths_traces = np.mean(recon_data.raw_data.T[roi_mask.T].T, axis=2)
        self.wavelengths_traces = wavelengths_traces
        for i, wl in enumerate(self.pa_data.get_wavelengths()):
            self.pa_data_table[wl] = self.wavelengths_traces[:, i]

    def get_overall_data_table(self, average="both"):
        new_df = {}

        po2_time = self.po2_data["Time"].to_numpy().astype(np.float)
        pa_time = self.pa_data_table["Time"].to_numpy().astype(np.float)

        df_time = np.linspace(0, np.max(po2_time), po2_time.shape[0])

        pa_so2 = self.pa_data_table["so2_pa"].to_numpy()
        if average == "both":
            probe_so2 = (self.po2_data["so2 (Pre)"] + self.po2_data["so2 (Post)"]) / 2
        else:
            probe_so2 = self.po2_data[f"so2 ({average})"]

        new_df["so2_pa"] = np.interp(df_time, pa_time, pa_so2) * 100
        new_df["so2_probe"] = np.interp(df_time, po2_time, probe_so2)

        pa_thb = self.pa_data_table["thb"].to_numpy()
        new_df["thb"] = np.interp(df_time, pa_time, pa_thb)
        new_df["Time"] = df_time

        # Align the curves:

        pa_so2_gradient = np.gradient(new_df["so2_pa"])
        probe_so2_gradient = np.gradient(new_df["so2_probe"])

        lags = signal.correlation_lags(len(pa_so2_gradient), len(probe_so2_gradient), "full")
        correlation = signal.correlate(pa_so2_gradient, probe_so2_gradient, "full")
        correlation[np.abs(lags) > 2000] = 0
        shift = lags[np.argmax(correlation)]
        # remove shift
        new_df["Time"] = new_df["Time"][:-np.abs(shift)]
        if shift > 0:
            new_df["so2_pa"] = new_df["so2_pa"][shift:]
            new_df["thb"] = new_df["thb"][shift:]
            new_df["so2_probe"] = new_df["so2_probe"][:-shift]
        else:
            new_df["so2_pa"] = new_df["so2_pa"][:shift]
            new_df["thb"] = new_df["thb"][:shift]
            new_df["so2_probe"] = new_df["so2_probe"][-shift:]

        for i, wl in enumerate(self.pa_data.get_wavelengths()):
            new_df[wl] = np.interp(df_time, pa_time, self.pa_data_table[wl])
            if shift > 0:
                new_df[wl] = new_df[wl][shift:]
            else:
                new_df[wl] = new_df[wl][:shift]
        new_df["Time"] /= 1e9
        new_df = pd.DataFrame(new_df)
        so2s = new_df["so2_pa"]
        import hampel
        outliers = hampel.hampel(so2s, window_size=100)
        new_df.loc[outliers] = np.nan
        return new_df