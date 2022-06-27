import h5py
import numpy as np


indices = np.arange(3248)   # list [0,.....,42775]
shuffled_indices = np.random.permutation(indices)[:1286]
f=h5py.File('./BOT_3248_20_28_28_3.h5','r')
data1=f['data'][:].reshape(3248,-1)[shuffled_indices]
f.close()

indices = np.arange(15029)   # list [0,.....,42775]
shuffled_indices = np.random.permutation(indices)[:5948]
f=h5py.File('./HS_15029_20_28_28_3.h5','r')
data2=f['data'][:].reshape(15029,-1)[shuffled_indices]
f.close()

indices = np.arange(5211)   # list [0,.....,42775]
shuffled_indices = np.random.permutation(indices)[:2062]
f=h5py.File('./KSC_5211_20_28_28_3.h5','r')
data3=f['data'][:].reshape(5211,-1)[shuffled_indices]
f.close()

indices = np.arange(30704)   # list [0,.....,42775]
shuffled_indices = np.random.permutation(indices)
f=h5py.File('./CH_30704_20_28_28_3.h5','r')
data4=f['data'][:].reshape(30704,-1)[shuffled_indices]
f.close()


print(data1.shape) # (3600, 8100) 18
print(data2.shape) # (3200, 8100) 16
print(data3.shape) # (1800, 28900) 9
print(data4.shape)


data=np.vstack((data1,data2,data3,data4))
print(data.shape) # (8600, 28900) 8200 = 43*200
#data = data.reshape(800, 100, 28, 28, 3)

indices = np.arange(data.shape[0])
shuffled_indices = np.random.permutation(indices)
data = data[shuffled_indices]
print(data.shape)

f=h5py.File('./train_' + str(data.shape[0]) + '_' + str(data.shape[1]) + '.h5','w')
f['data']=data # (20000, 47040)
f.close()
