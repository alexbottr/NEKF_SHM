import h5py
import numpy as np
import pickle

def traverse_group_first_line(group, prefix=""):
    data_list = []

    for key in group.keys():
        if "healthy" in key:
            continue
        
        elif isinstance(group[key], h5py.Dataset):
            dataset = group[key][()]  # Obtenez les données du dataset
            Y = np.zeros((75, 1))
            U = np.zeros((75, 0))
            #Y[0, :] = [0.0, 0.0, 0.0, 0.0, 0.0]
            Y[:, 0] = dataset[2][:150:2] * 10
            #Y[:, 1] = dataset[3][:5000:5] * 10
            #Y[:, 2] = dataset[4][:5000:5] * 10
            #Y[:, 3] = dataset[5][:5000:5] * 10
            #Y[:, 4] = dataset[6][:5000:5] * 10
            data_list.append((dataset[0][:150:2], Y, U))
            continue

        elif isinstance(group[key], h5py.Group):
            # Assurez-vous de fournir le bon préfixe lors de la récursion
            data_list.extend(traverse_group_first_line(group[key], prefix=f"{prefix}/{key}"))

    return data_list

def find_dam_data(group):
    dam_data_list = []
    for key in group.keys():
        if key == "healthy":
            dam_data_list.extend(traverse_group_first_line(group[key]))
    return dam_data_list

# Ouverture du fichier HDF5 en utilisant le gestionnaire de contexte pour s'assurer qu'il est fermé correctement
with h5py.File(r'Z:\test_doe3.h5', 'r') as traj_file:
    dam_data_list = find_dam_data(traj_file)
    data_list = traverse_group_first_line(traj_file)

print(dam_data_list)
print(len(data_list))
print(data_list[0][1].shape)

with open('white_noise_train_ensam4.pickle', 'wb') as handle:
    pickle.dump(data_list, handle)

with open('white_noise_val_ensam4.pickle', 'wb') as handle:
    pickle.dump(dam_data_list, handle)