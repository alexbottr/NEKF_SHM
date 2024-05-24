import numpy as np
import scipy.io as sio
import pickle

data = []
for i in range(2, 21):
    transposed_data = np.transpose(sio.loadmat(f"Z:\\Mat\\full_whitenoise_8ch_2.mat")[f"whitenoise{i}e"])
    data.append(transposed_data)
#data.append(np.transpose(sio.loadmat(f"Z:\\Mat\\full_whitenoise.mat")[f"whitenoise20d"]))
L_train = []
for i in range(len(data)):
    #T = []
    Y = np.zeros((20000, 1))
    U = np.zeros((20000, 1))
    T=np.arange(0, 12, 6.0e-04)
    Y[:, 0] = data[i][1][:20000]
    #Y[:, 1] = data[i][2]
    #Y[:, 2] = data[i][3]
    #Y[:, 3] = data[i][4]
    #Y[:, 4] = data[i][5]
    #Y[:, 5] = data[i][6]
    #Y[:, 6] = data[i][7]
    #Y[:, 7] = data[i][8]
    U[:, 0] = data[i][9][:20000]
    L_train.append((T, Y, U))

transposed_data_val = np.transpose(sio.loadmat(f"Z:\\Mat\\full_whitenoise_8ch_2.mat")["whitenoise1e"])
L_val=[]
#T2 = []
Y2 = np.zeros((20000, 1))
U2 = np.zeros((20000, 1))
T2=np.arange(0, 12, 6.0e-04)
Y2[:, 0] = transposed_data_val[1][:20000]
#Y2[:, 1] = transposed_data_val[2]
#Y2[:, 2] = transposed_data_val[3]
#Y2[:, 3] = transposed_data_val[4]
#Y2[:, 4] = transposed_data_val[5]
#Y2[:, 5] = transposed_data_val[6]
#Y2[:, 6] = transposed_data_val[7]
#Y2[:, 7] = transposed_data_val[8]
U2[:, 0] = transposed_data_val[9][:20000]
L_val.append((T2, Y2, U2))



print(((L_val)))



with open('SHM_train_1channel_6.pickle', 'wb') as handle:
    pickle.dump(L_train, handle)

with open('SHM_val_1channel_6.pickle', 'wb') as handle:
    pickle.dump(L_val, handle)
