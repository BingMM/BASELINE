#%% Import

from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from apexpy import Apex

#%% Paths

base = '/home/bing/Dropbox/work/code/repos/BASELINE'
base_data = pjoin(base, 'data')
base_figure = pjoin(base, 'figures')

#%% Generate synthetic data

t_start = datetime(2000, 1, 1)
t_duration = 2*365*24*60
t = np.array([t_start + timedelta(seconds=i*60) for i in range(t_duration)])

glat, glon = 65, 0
mlat = np.zeros(t.size)
s_since_2000 = np.zeros(t.size).astype(int)
for i, ti in tqdm(enumerate(t), total=t.size, desc='Generate mlat and t'):
    apx = Apex(ti.year, refh=0)
    mlat[i] = apx.convert(glat, glon, 'geo', 'apex', height=0)[0]
    s_since_2000[i] = (ti - datetime(2000, 1, 1)).total_seconds()

Be = 5 * np.random.normal(0, 1, t.size)
Bn = 5 * np.random.normal(0, 1, t.size)
Bu = 5 * np.random.normal(0, 1, t.size)

q = np.arange(24*60)
daily = 15e3 / (150 * np.sqrt(2)) * np.exp(-.5 * ((q-576)**2 / (150**2)))
daily = np.tile(daily, 2*365)

yearly = 5*np.sin(np.arange(t.size)/(365*24*60)*np.pi)

offset = 35000

Be_comp = Be + daily + yearly + offset
Bn_comp = Bn + daily + yearly + offset
Bu_comp = Bu + daily + yearly + offset

#%% Plot data

points = 10*24*60
t_plot = t[:points]

fig, axs = plt.subplots(4, 3, figsize=(15, 10), sharex=True)

axs[0,0].plot(t_plot, Be_comp[:points])
axs[0,0].set_title('Be')
axs[0,1].plot(t_plot, Bn_comp[:points])
axs[0,1].set_title('Bn')
axs[0,2].plot(t_plot, Bu_comp[:points])
axs[0,2].set_title('Bu')

axs[1,0].plot(t_plot, Be[:points])
axs[1,0].set_title('Signal')
axs[1,1].plot(t_plot, Bn[:points])
axs[1,1].set_title('Signal')
axs[1,2].plot(t_plot, Bu[:points])
axs[1,2].set_title('Signal')

axs[2,0].plot(t_plot, daily[:points])
axs[2,0].set_title('Daily variation')
axs[2,1].plot(t_plot, daily[:points])
axs[2,1].set_title('Daily variation')
axs[2,2].plot(t_plot, daily[:points])
axs[2,2].set_title('Daily variation')

axs[3,0].plot(t_plot, yearly[:points])
axs[3,0].set_title('Yearly variation')
axs[3,1].plot(t_plot, yearly[:points])
axs[3,1].set_title('Yearly variation')
axs[3,2].plot(t_plot, yearly[:points])
axs[3,2].set_title('Yearly variation')

filename = pjoin(base_figure, 'synthetic_data.png')
plt.savefig(filename, bbox_inches='tight')
plt.close('all')

#%% Save synthetic data

np.save(pjoin(base_data, 's_since_2000.npy'), s_since_2000)
np.save(pjoin(base_data, 'mlat.npy'), mlat)

np.save(pjoin(base_data, 'Be_truth.npy'), Be)
np.save(pjoin(base_data, 'Bn_truth.npy'), Bn)
np.save(pjoin(base_data, 'Bu_truth.npy'), Bu)

np.save(pjoin(base_data, 'Be.npy'), Be_comp)
np.save(pjoin(base_data, 'Bn.npy'), Bn_comp)
np.save(pjoin(base_data, 'Bu.npy'), Bu_comp)

