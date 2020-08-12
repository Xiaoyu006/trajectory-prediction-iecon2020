import numpy as np
HIST = np.load('Hist_LC.npy')
FUT = np.load('Fut_LC.npy')
print(HIST.shape)
print(FUT.shape)
np.random.seed(1)
state = np.random.get_state()
np.random.shuffle(HIST)
np.random.set_state(state)
np.random.shuffle(FUT)

train_size = int(0.70 * HIST.shape[0])
val_size = HIST.shape[0] - train_size

np.save('Hist_LC_train', HIST[:train_size])
np.save('Fut_LC_train', FUT[:train_size])
np.save('Hist_LC_test', HIST[train_size:])
np.save('Fut_LC_test', FUT[train_size:])

print(HIST[:train_size].shape)
print(FUT[:train_size].shape)
print(HIST[train_size:].shape)
print(FUT[train_size:].shape)