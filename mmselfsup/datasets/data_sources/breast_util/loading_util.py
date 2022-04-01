import numpy as np
import h5py

def read_h5(file_name):
    data = h5py.File(file_name, 'r')
    image = np.array(data['image']).T
    data.close()
    return image