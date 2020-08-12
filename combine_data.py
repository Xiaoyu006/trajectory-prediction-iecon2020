import numpy as np
Hist_LC = np.empty([0,9,31,2])
Fut_LC = np.empty([0,1,50,2])

for tt in ['0750', '0805','0820']:
    h = np.load('HIST_{}amlc_selection.npy'.format(tt))
    f = np.load('FUT_{}amlc_selection.npy'.format(tt))
    print(h.shape)
    Hist_LC = np.concatenate((Hist_LC, h), axis=0) #np.concatenate((a, b.T), axis=1)
    Fut_LC = np.concatenate((Fut_LC, f), axis=0) #np.concatenate((a, b.T), axis=1)                   
    
print(Hist_LC.shape)
print(Fut_LC.shape)

np.save('Hist_LC', Hist_LC)
np.save('Fut_LC', Fut_LC)