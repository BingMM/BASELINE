#%% Import

from os.path import join as pjoin
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from baseline import BaselineEstimator, VarianceEstimator

#%% Paths

base = '/home/bing/Dropbox/work/code/repos/BASELINE'
base_data = pjoin(base, 'data')
base_figure = pjoin(base, 'figures')

#%% Import data

Be_truth        = np.load(pjoin(base_data, 'Be_truth.npy'))
Bn_truth        = np.load(pjoin(base_data, 'Bn_truth.npy'))
Bu_truth        = np.load(pjoin(base_data, 'Bu_truth.npy'))

Be              = np.load(pjoin(base_data, 'Be.npy'))
Bn              = np.load(pjoin(base_data, 'Bn.npy'))
Bu              = np.load(pjoin(base_data, 'Bu.npy'))

mlat            = np.load(pjoin(base_data, 'mlat.npy'))

s_since_2000    = np.load(pjoin(base_data, 's_since_2000.npy'))
t = np.array([datetime(2000, 1, 1) + timedelta(seconds=float(s)) for s in s_since_2000])
del s_since_2000

#%% Estimate variance

VE = VarianceEstimator(t, Bn, Be, Bu, mlat)
VE.estimate()

#%% Estimate baseline - N

BE_N = BaselineEstimator(t, Bn, VE.df['uN'].values, mlat, component='N')
BE_N.get_baseline()

#%% Step 1 a

plt.figure()
plt.plot(BE_N.df['datetime'][:7*1440], BE_N.df['x'][:7*1440])
plt.plot(BE_N.QD_step_1a[:7], '.')

#%% Step 1 b

fig, axs = plt.subplots(2, 1)
axs[0].plot(BE_N.df['datetime'][:7*1440], BE_N.df['x'][:7*1440])
axs[0].plot(BE_N.df['datetime'][:7*1440], BE_N.df['step_1b'][:7*1440])

axs[1].plot(BE_N.df['datetime'][:7*1440], BE_N.df['residual_step_1'][:7*1440])

#%% Step 1 c

plt.figure()
plt.plot(BE_N.df['datetime'][5*1440:7*1440], BE_N.df['residual_step_1'][5*1440:7*1440])
plt.plot(BE_N.QD_step_1c[5*48:7*48], '.')

#%% Step 1 d

plt.figure()
plt.plot(BE_N.df['datetime'][5*1440:7*1440], BE_N.df['residual_step_1'][5*1440:7*1440])
plt.plot(BE_N.QD_step_1c[5*48:7*48], '.')
plt.plot(BE_N.df['datetime'][5*1440:7*1440], BE_N.df['QD'][5*1440:7*1440])

#%% Step 1 e

plt.figure()
plt.plot(BE_N.df['datetime'][5*1440:7*1440], BE_N.df['x'][5*1440:7*1440])
plt.plot(BE_N.df['datetime'][5*1440:7*1440], BE_N.df['x_QD'][5*1440:7*1440])

plt.figure()
plt.plot(BE_N.df['datetime'][5*1440:7*1440], (Bn_truth + 35000)[5*1440:7*1440])
plt.plot(BE_N.df['datetime'][5*1440:7*1440], BE_N.df['x_QD'][5*1440:7*1440])

#%% Step 2.a

plt.figure()
plt.plot(BE_N.df['datetime'][0*1440:50*1440], BE_N.df['x_QD'][0*1440:50*1440])
plt.plot(BE_N.QD_step_2a[:50], '.')

#%% Step 2.b

plt.figure()
plt.plot(BE_N.df['datetime'][0*1440:500*1440], BE_N.df['x_QD'][0*1440:500*1440])
plt.plot(BE_N.df['datetime'][0*1440:500*1440], BE_N.df['QY'][0*1440:500*1440])

#%% Step 2.c

fig, axs = plt.subplots(2, 1)
axs[0].plot(BE_N.df['datetime'][0*1440:500*1440], BE_N.df['x_QD'][0*1440:500*1440])
axs[1].plot(BE_N.df['datetime'][0*1440:500*1440], BE_N.df['x_QD_QY'][0*1440:500*1440])

plt.figure()
plt.plot(BE_N.df['datetime'][5*1440:5*1440+200], Bn_truth[5*1440:5*1440+200])
plt.plot(BE_N.df['datetime'][5*1440:5*1440+200], BE_N.df['x_QD_QY'][5*1440:5*1440+200])
