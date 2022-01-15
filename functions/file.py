import h5py
import os.path


def save_file(folder,name,data,xaxis,yaxis):
    path = folder + '/' + name
    if not os.path.exists(path):
        with h5py.File(path, 'w') as f:
            f.create_dataset('data', data = data)
            f.create_dataset('X_'+xaxis, data = 0)
            f.create_dataset('Y_'+yaxis, data = 0)
            f.close()
            print('The file has been saved')
    else:
        print('The file already exists')
    return None


def rewrite_file(folder,name,data,xaxis,yaxis):
    path = folder + '/' + name
    if os.path.exists(path):
        with h5py.File(path, 'w') as f:
            f.create_dataset('data', data = data)
            f.create_dataset('X_'+xaxis, data = 0)
            f.create_dataset('Y_'+yaxis, data = 0)
            f.close()
            print('The file has been rewrited')
    else:
        print('The file does not exist')
    return None


def read_file(folder,name):
    path = folder + '/' + name
    with h5py.File(path, 'r') as f:
        data = f['data'][:]
        print(f.keys())
        f.close()
    print('Data shape =',data.shape)
    return data