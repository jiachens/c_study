import os
import sys
import glob
import h5py
import numpy as np
np.random.seed(666)
from torch.utils.data import Dataset
import torch
torch.manual_seed(666)
torch.cuda.manual_seed_all(666)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

sys.path.append('./latent_3d_points_py3/')
from latent_3d_points_py3.src import in_out
from latent_3d_points_py3.src.general_utils import plot_3d_point_cloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
SHAPENET_DIR = './data/shape_net_core_uniform_samples_2048/'
scratch_shapenet_dir = '/scratch/shape_net_core_uniform_samples_2048'
if os.path.exists(scratch_shapenet_dir):
    SHAPENET_DIR = scratch_shapenet_dir
    print(f'Loading shapenet from {SHAPENET_DIR}')

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip -qq %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

    # if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    #     www = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
    #     zipfile = os.path.basename(www)
    #     os.system('wget %s --no-check-certificate; unzip %s -q' % (www, zipfile))
    #     os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    #     os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'PointDA_data')):
        os.system('python gdrivedl.py https://drive.google.com/file/d/1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J/view\\?usp\\=sharing -P ./data -q')
        os.system('unzip -qq ./data/PointDA_data.zip')

    if not os.path.exists(os.path.join(DATA_DIR, 'ScanObjectNN')):
        os.system('mkdir ./data/ScanObjectNN')
        os.system('wget --no-check-certificate \'https://docs.google.com/uc?export=download&id=1ZeqaYNp6wWL_xKvuZ5onZbChlI9UZYva\' -O train.h5')
        os.system('mv train.h5 ./data/ScanObjectNN/')
        os.system('wget --no-check-certificate \'https://docs.google.com/uc?export=download&id=14zIWTP0jq911TqlT4nfkxLlFLW-mCnoh\' -O test.h5')
        os.system('mv test.h5 ./data/ScanObjectNN/')


def parse_dataset_modelnet10(partition,num_points=1024):
    download()
    # DATA_DIR = tf.keras.utils.get_file(
    #     "modelnet.zip",
    #     "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    #     extract=True,
    #     cache_dir=
    # )
    # DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")
    DATA_DIR = './data/PointDA_data/modelnet'

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    index = np.random.choice(2048, 1024, replace=False)

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        if partition == 'train':
            train_files = glob.glob(os.path.join(folder, "train/*"))
        elif partition == 'test':
            test_files = glob.glob(os.path.join(folder, "test/*"))
        if partition == 'train':
            for f in train_files:
                raw = np.load(f)[index,:]
                # print(np.max(raw),np.min(raw))
                # mean_x, mean_y, mean_z = np.mean(raw[:,0]), np.mean(raw[:,1]), np.mean(raw[:,2])
                # raw[:,0] -= mean_x
                # raw[:,1] -= mean_y
                # raw[:,2] -= mean_z
                leng_x, leng_y, leng_z = np.max(raw[:,0]) - np.min(raw[:,0]), np.max(raw[:,1]) - np.min(raw[:,1]), np.max(raw[:,2]) - np.min(raw[:,2])
                raw[:,0] -= (np.max(raw[:,0]) + np.min(raw[:,0])) / 2
                raw[:,1] -= (np.max(raw[:,1]) + np.min(raw[:,1])) / 2
                raw[:,2] -= (np.max(raw[:,2]) + np.min(raw[:,2])) / 2
                if leng_x >= leng_y and leng_x >= leng_z:
                    ratio = 2.0 / leng_x
                elif leng_y >= leng_x and leng_y >= leng_z:
                    ratio = 2.0 / leng_y
                else:
                    ratio = 2.0 / leng_z

                raw *= ratio
                train_points.append(raw)
                train_labels.append(i)


        elif partition == 'test':
            for f in test_files:
                raw = np.load(f)[index,:]
                leng_x, leng_y, leng_z = np.max(raw[:,0]) - np.min(raw[:,0]), np.max(raw[:,1]) - np.min(raw[:,1]), np.max(raw[:,2]) - np.min(raw[:,2])
                raw[:,0] -= (np.max(raw[:,0]) + np.min(raw[:,0])) / 2
                raw[:,1] -= (np.max(raw[:,1]) + np.min(raw[:,1])) / 2
                raw[:,2] -= (np.max(raw[:,2]) + np.min(raw[:,2])) / 2
                if leng_x >= leng_y and leng_x >= leng_z:
                    ratio = 2.0 / leng_x
                elif leng_y >= leng_x and leng_y >= leng_z:
                    ratio = 2.0 / leng_y
                else:
                    ratio = 2.0 / leng_z

                raw *= ratio

                test_points.append(raw)
                test_labels.append(i)

    if partition == 'train':
        return np.array(train_points), np.array(train_labels)
    elif partition == 'test':
        return np.array(test_points), np.array(test_labels)

    # np.save('./data/ModelNet10/train_points.npy',np.array(train_points))
    # np.save('./data/ModelNet10/test_points.npy',np.array(test_points))
    # np.save('./data/ModelNet10/train_labels.npy',np.array(train_labels))
    # np.save('./data/ModelNet10/test_labels.npy',np.array(test_labels))

def parse_dataset_shapenet10(partition,num_points=1024):
    download()
    DATA_DIR = './data/PointDA_data/shapenet'

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    index = np.random.choice(2048, 1024, replace=False)

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        if partition == 'train':
            train_files = glob.glob(os.path.join(folder, "train/*"))
        elif partition == 'test':
            test_files = glob.glob(os.path.join(folder, "test/*"))
        if partition == 'train':
            for f in train_files:
                raw = np.load(f)[index,:]
                leng_x, leng_y, leng_z = np.max(raw[:,0]) - np.min(raw[:,0]), np.max(raw[:,1]) - np.min(raw[:,1]), np.max(raw[:,2]) - np.min(raw[:,2])
                raw[:,0] -= (np.max(raw[:,0]) + np.min(raw[:,0])) / 2
                raw[:,1] -= (np.max(raw[:,1]) + np.min(raw[:,1])) / 2
                raw[:,2] -= (np.max(raw[:,2]) + np.min(raw[:,2])) / 2
                if leng_x >= leng_y and leng_x >= leng_z:
                    ratio = 2.0 / leng_x
                elif leng_y >= leng_x and leng_y >= leng_z:
                    ratio = 2.0 / leng_y
                else:
                    ratio = 2.0 / leng_z

                raw *= ratio

                train_points.append(raw)
                train_labels.append(i)
            
        elif partition == 'test':
            for f in test_files:
                raw = np.load(f)[index,:]
                leng_x, leng_y, leng_z = np.max(raw[:,0]) - np.min(raw[:,0]), np.max(raw[:,1]) - np.min(raw[:,1]), np.max(raw[:,2]) - np.min(raw[:,2])
                raw[:,0] -= (np.max(raw[:,0]) + np.min(raw[:,0])) / 2
                raw[:,1] -= (np.max(raw[:,1]) + np.min(raw[:,1])) / 2
                raw[:,2] -= (np.max(raw[:,2]) + np.min(raw[:,2])) / 2
                if leng_x >= leng_y and leng_x >= leng_z:
                    ratio = 2.0 / leng_x
                elif leng_y >= leng_x and leng_y >= leng_z:
                    ratio = 2.0 / leng_y
                else:
                    ratio = 2.0 / leng_z

                raw *= ratio

                test_points.append(raw)
                test_labels.append(i)

    if partition == 'train':
        return np.array(train_points), np.array(train_labels)
    elif partition == 'test':
        return np.array(test_points), np.array(test_labels)

    # np.save('./data/ModelNet10/train_points.npy',np.array(train_points))
    # np.save('./data/ModelNet10/test_points.npy',np.array(test_points))
    # np.save('./data/ModelNet10/train_labels.npy',np.array(train_labels))
    # np.save('./data/ModelNet10/test_labels.npy',np.array(test_labels))


def parse_dataset_scanobject(partition,num_points=1024):
    download()
    # DATA_DIR = tf.keras.utils.get_file(
    #     "modelnet.zip",
    #     "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    #     extract=True,
    #     cache_dir=
    # )
    # DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")
    DATA_DIR = './data/ScanObjectNN'


    index = np.random.choice(2048, 1024, replace=False)

    if partition == 'train':

        f = h5py.File(os.path.join(DATA_DIR, "train.h5"))
        train_data = f['data'][:][:,index,:]
        for i in range(train_data.shape[0]):
            # mean_x, mean_y, mean_z = np.mean(train_data[i,:,0]), np.mean(train_data[i,:,1]), np.mean(train_data[i,:,2])
            train_data[i,:,0] -= (np.max(train_data[i,:,0]) + np.min(train_data[i,:,0])) / 2
            train_data[i,:,1] -= (np.max(train_data[i,:,1]) + np.min(train_data[i,:,1])) / 2
            train_data[i,:,2] -= (np.max(train_data[i,:,2]) + np.min(train_data[i,:,2])) / 2
            leng_x, leng_y, leng_z = np.max(train_data[i,:,0]) - np.min(train_data[i,:,0]), np.max(train_data[i,:,1]) - np.min(train_data[i,:,1]), np.max(train_data[i,:,2]) - np.min(train_data[i,:,2])
            if leng_x >= leng_y and leng_x >= leng_z:
                ratio = 2.0 / leng_x
            elif leng_y >= leng_x and leng_y >= leng_z:
                ratio = 2.0 / leng_y
            else:
                ratio = 2.0 / leng_z

            train_data[i,:,:] *= ratio

        train_label = f['label'][:]
        return np.array(train_data), np.array(train_label)

    elif partition == 'test':

        f = h5py.File(os.path.join(DATA_DIR, "test.h5"))
        test_data = f['data'][:][:,index,:]
        for i in range(test_data.shape[0]):
            # mean_x, mean_y, mean_z = np.mean(test_data[i,:,0]), np.mean(test_data[i,:,1]), np.mean(test_data[i,:,2])
            test_data[i,:,0] -= (np.max(test_data[i,:,0]) + np.min(test_data[i,:,0])) / 2
            test_data[i,:,1] -= (np.max(test_data[i,:,1]) + np.min(test_data[i,:,1])) / 2
            test_data[i,:,2] -= (np.max(test_data[i,:,2]) + np.min(test_data[i,:,2])) / 2
            leng_x, leng_y, leng_z = np.max(test_data[i,:,0]) - np.min(test_data[i,:,0]), np.max(test_data[i,:,1]) - np.min(test_data[i,:,1]), np.max(test_data[i,:,2]) - np.min(test_data[i,:,2])
            if leng_x >= leng_y and leng_x >= leng_z:
                ratio = 2.0 / leng_x
            elif leng_y >= leng_x and leng_y >= leng_z:
                ratio = 2.0 / leng_y
            else:
                ratio = 2.0 / leng_z

            test_data[i,:,:] *= ratio
        
        test_label = f['label'][:]
        return np.array(test_data), np.array(test_label)

    # np.save('./data/ScanObjectNN/train_points.npy',np.array(train_data))
    # np.save('./data/ScanObjectNN/test_points.npy',np.array(test_data))
    # np.save('./data/ScanObjectNN/train_labels.npy',np.array(train_label))
    # np.save('./data/ScanObjectNN/test_labels.npy',np.array(test_label))

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx]

def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_data, all_label = shuffle_data(all_data, all_label)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud,xyz1,xyz2


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    jitter = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return (pointcloud + jitter).astype('float32'),jitter.astype('float32') 

# class ATTA_ModelNet40(Dataset):
#     def __init__(self, num_points):
#         self.data, self.label = load_data('train')
#         self.num_points = num_points
        
#     def __getitem__(self, item):
#         pointcloud = self.data[item][:self.num_points]
#         label = self.label[item]
#         pointcloud,jitter=jitter_pointcloud(pointcloud)
#         idx = np.arange(pointcloud.shape[0])
#         np.random.shuffle(idx)
#         transform=(jitter,idx)
#         return pointcloud[idx], label,item,transform

#     def __len__(self):
#         return self.data.shape[0]

def rotate_data(data, label):
    """ Rotate a batch of points by the label
        Input:
          BxNx3 array
        Return:
          BxNx3 array
        
    """
    rotate_func = rotate_point_cloud_by_angle_xyz
    # batch_size = batch_data.shape[0]
    rotated_data = np.zeros(data.shape, dtype=np.float32)
    # for k in range(batch_size):
    shape_pc = data
    l = label
    if l==0:
        pass
    elif 1<=l<=3:
        shape_pc = rotate_func(shape_pc, angle_x=l*np.pi/2)
    elif 4<=l<=5:
        shape_pc = rotate_func(shape_pc, angle_z=(l*2-7)*np.pi/2)
    elif 6<=l<=9:
        shape_pc = rotate_func(shape_pc, angle_x=(l*2-11)*np.pi/4)
    elif 10<=l<=13:
        shape_pc = rotate_func(shape_pc, angle_z=(l*2-19)*np.pi/4)
    else: #l == 14 ~ 17
        shape_pc = rotate_func(shape_pc, angle_x=np.pi/2, angle_z=(l*2-27)*np.pi/4)
    rotated_data = shape_pc

    return rotated_data


def rotate_point_cloud_by_angle_xyz(data, angle_x=0, angle_y=0, angle_z=0):
    """ Rotate the point cloud along up direction with certain angle.
        Rotate in the order of x, y and then z.

    """
    rotated_data = data.reshape((-1, 3))
    
    cosval = np.cos(angle_x)
    sinval = np.sin(angle_x)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    rotated_data = np.dot(rotated_data, rotation_matrix)

    cosval = np.cos(angle_y)
    sinval = np.sin(angle_y)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(rotated_data, rotation_matrix)

    cosval = np.cos(angle_z)
    sinval = np.sin(angle_z)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(rotated_data, rotation_matrix)
    
    return rotated_data.reshape(data.shape)

class PCData(Dataset):
    def __init__(self, num_points, name='modelnet40' ,partition='train', translate = False, jitter=True, rotation=False, angles=6):
        
        if name == 'modelnet40':
            self.data, self.label = load_data(partition)
        elif name == 'modelnet10':
            self.data, self.label = parse_dataset_modelnet10(partition)
        elif name == 'shapenet10':
            self.data, self.label = parse_dataset_shapenet10(partition)
        elif name == 'scanobjectnn':
            self.data, self.label = parse_dataset_scanobject(partition)

        self.num_points = num_points
        self.partition = partition
        self.rotation = rotation
        self.angles = angles
        self.jitter = jitter
        self.translate = translate

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if not self.rotation:
            if self.partition == 'train':
                if self.jitter:
                    pointcloud,_ = jitter_pointcloud(pointcloud)
                if self.translate:
                    pointcloud,_,_ = translate_pointcloud(pointcloud)
                # np.random.shuffle(pointcloud)
            return pointcloud.astype('float32'), label, pointcloud.astype('float32'), label # the latter two are not used
        else:
            if self.partition == 'train':
                if self.jitter:
                    pointcloud,_ = jitter_pointcloud(pointcloud)
                if self.translate:
                    pointcloud,_,_ = translate_pointcloud(pointcloud)
                # np.random.shuffle(pointcloud)
            rotation_label = np.random.randint(self.angles)
            #rotation_label = np.squeeze(rotation_label)
            rotated_pointcloud = rotate_data(pointcloud, rotation_label)

            return pointcloud.astype('float32'),label,rotated_pointcloud.astype('float32'),rotation_label


    def __len__(self):
        return self.data.shape[0]

def generate_jigsaw_data_label(pointcloud, k):

    jigsaw_pointcloud = []
    jigsaw = np.random.permutation(k**3)
    label = []
    
    interval = np.linspace(-1,1,k+1)
    interval[0] = -1.05
    interval[-1] = 1.06

    for i in range(k):
        for j in range(k):
            for p in range(k):
                idx = np.argwhere((pointcloud[:,0] >= interval[i]) & (pointcloud[:,0] < interval[i+1])
                        & (pointcloud[:,1] >= interval[j]) &  (pointcloud[:,1] < interval[j+1])
                        & (pointcloud[:,2] >= interval[p]) &  (pointcloud[:,2] < interval[p+1]))
                idx = np.squeeze(idx)

                jigsaw_id = jigsaw[i * k**2 + j * k**1 + p]
                temp = pointcloud[idx,:].reshape(-1,3)
        
                jigsaw_i = jigsaw_id // (k**2) 
                jigsaw_j = (jigsaw_id - jigsaw_i * (k**2)) // k**1
                jigsaw_p = jigsaw_id - jigsaw_i * (k**2) - jigsaw_j * (k**1)

                temp[:,0] += (jigsaw_i - i) * (2. / k)
                temp[:,1] += (jigsaw_j - j) * (2. / k)
                temp[:,2] += (jigsaw_p - p) * (2. / k)

                temp_label = np.ones(temp.shape[0]) * (i * k**2 + j * k**1 + p)
                jigsaw_pointcloud.append(temp)
                label.append(temp_label)

    jigsaw_pointcloud = np.concatenate(jigsaw_pointcloud)
    label = np.concatenate(label)

    jigsaw_pointcloud,label = shuffle_data(jigsaw_pointcloud,label)
    # print(jigsaw_pointcloud.shape)

    return jigsaw_pointcloud, label


class PCData_SSL(Dataset):
    def __init__(self, num_points, name='modelnet40', partition='train', rotation=False, angles=6, jigsaw=False, k=2):
        if name == 'modelnet40':
            self.data, self.label = load_data(partition)
        elif name == 'modelnet10':
            self.data, self.label = parse_dataset_modelnet10(partition)
        elif name == 'shapenet10':
            self.data, self.label = parse_dataset_shapenet10(partition)
        elif name == 'scanobjectnn':
            self.data, self.label = parse_dataset_scanobject(partition)
        self.num_points = num_points
        self.partition = partition
        self.jigsaw = jigsaw
        self.k = k
        self.rotation = rotation
        self.angles = angles

        # print(np.max(self.data), np.min(self.data))


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if not self.jigsaw and not self.rotation:
            if self.partition == 'train':
                pointcloud,_ = jitter_pointcloud(pointcloud)
                # np.random.shuffle(pointcloud)
            return pointcloud, label # the latter two are not used 
        elif self.jigsaw:

            if self.partition == 'train':
                pointcloud,_ = jitter_pointcloud(pointcloud)
                # np.random.shuffle(pointcloud)
            jigsaw_pointcloud, jigsaw_label = generate_jigsaw_data_label(pointcloud,self.k)
            #rotation_label = np.squeeze(rotation_label)

            return jigsaw_pointcloud.astype('float32'),jigsaw_label
        elif self.rotation:

            if self.partition == 'train':
                pointcloud,_ = jitter_pointcloud(pointcloud)
                # np.random.shuffle(pointcloud)
            rotation_label = np.random.randint(self.angles)
            #rotation_label = np.squeeze(rotation_label)
            rotated_pointcloud = rotate_data(pointcloud, rotation_label)

            return rotated_pointcloud.astype('float32'),rotation_label


    def __len__(self):
        return self.data.shape[0]

class PCData_Jigsaw(Dataset):
    def __init__(self, num_points, name='modelnet40', partition='train', jigsaw=False, k=2):
        
        if name == 'modelnet40':
            self.data, self.label = load_data(partition)
        elif name == 'modelnet10':
            self.data, self.label = parse_dataset_modelnet10(partition)
        elif name == 'shapenet10':
            self.data, self.label = parse_dataset_shapenet10(partition)
        elif name == 'scanobjectnn':
            self.data, self.label = parse_dataset_scanobject(partition)
        self.num_points = num_points
        self.partition = partition
        self.jigsaw = jigsaw
        self.k = k



    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if not self.jigsaw:
            if self.partition == 'train':
                pointcloud,_ = jitter_pointcloud(pointcloud)
                # np.random.shuffle(pointcloud)
            return pointcloud.astype('float32'), label, pointcloud.astype('float32'), label # the latter two are not used 
        else:
            if self.partition == 'train':
                pointcloud,_ = jitter_pointcloud(pointcloud)
                # np.random.shuffle(pointcloud)
            jigsaw_pointcloud, jigsaw_label = generate_jigsaw_data_label(pointcloud,self.k)
            #rotation_label = np.squeeze(rotation_label)

            return pointcloud.astype('float32'),label,jigsaw_pointcloud.astype('float32'),jigsaw_label


    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    download()
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
