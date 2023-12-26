import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import random


class Myloader(DataLoader):
    def __init__(self, data_path, index_path, batch_size, num_workers, shuffle, pin_memory=False):
        self.data_path = data_path
        self.index_path = index_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        
        self.description = {
            'Atelectasis': 'float',
            'Calcification of the Aorta': 'float',
            'Cardiomegaly': 'float',
            'Consolidation': 'float',
            'Edema': 'float',
            'Enlarged Cardiomediastinum': 'float',
            'Fracture': 'float',
            'Lung Lesion': 'float',
            'Lung Opacity': 'float',
            'No Finding': 'float',
            'Pleural Effusion': 'float',
            'Pleural Other': 'float',
            'Pneumomediastinum': 'float',
            'Pneumonia': 'float',
            'Pneumoperitoneum': 'float',
            'Pneumothorax': 'float',
            'Subcutaneous Emphysema': 'float',
            'Support Devices': 'float',
            'Tortuous Aorta': 'float',
            'age': 'int',
            'dicom_id': 'byte',
            'gender': 'byte',
            'jpg_bytes': 'byte',
            'race': 'byte',
            'study_id': 'int',
            'subject_id': 'int',
        }
        
        if self.shuffle == True:
            self.dataset = TFRecordDataset(self.data_path, self.index_path, self.description, shuffle_queue_size = self.batch_size, transform = self.train_decode)
        else:
            self.dataset = TFRecordDataset(self.data_path, self.index_path, self.description, shuffle_queue_size = self.batch_size, transform = self.val_decode)
            
        g = torch.Generator()
        g.manual_seed(0)
             
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'worker_init_fn': self.seed_worker,
            'generator': g,
        }
        super().__init__(**self.init_kwargs)
        
    def train_decode(self, features):
        decode_img = cv2.imdecode(np.fromstring(features['jpg_bytes'], dtype=np.uint8), -1)
        features['jpg_bytes'] = Image.fromarray(decode_img).convert('RGB')
#         transform = transforms.Compose([
# #             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(15),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#         ])



        # 定义一系列的 transforms
        transforms_list = [
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(256, scale=(0.9, 1.0)),
            transforms.RandomPosterize(bits=6, p=0.2),
            transforms.RandomCrop(size=(256, 256), padding=(10, 10)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.001, 0.6))], p=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomVerticalFlip(p=0.5)
        ]

        # 随机选择 1 到 4 个 transforms
        num_transforms = random.randint(4, 6)
        selected_transforms = random.sample(transforms_list, num_transforms)

        # 创建 transform
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            *selected_transforms,
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])



        features['jpg_bytes'] = transform(features['jpg_bytes'])

        return features

    def val_decode(self, features):
        decode_img = cv2.imdecode(np.fromstring(features['jpg_bytes'], dtype=np.uint8), -1)
        features['jpg_bytes'] = Image.fromarray(decode_img).convert('RGB')
        transform = transforms.Compose([
#             transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        features['jpg_bytes'] = transform(features['jpg_bytes'])

        return features
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        
                
    @staticmethod    
    def collate_fn(data):
        imgs = []
        final_labels = []
        dicom_ids = []
        label_name = ['Atelectasis','Calcification of the Aorta','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax','Subcutaneous Emphysema','Support Devices','Tortuous Aorta']
        for example in data:
            labels = []
            imgs.append(example['jpg_bytes'])
            dicom_ids.append(example['dicom_id'])
            for name in label_name:
                labels.append(example[name])
            final_labels.append(labels)
        final_labels = np.array(final_labels)
            
        return torch.stack(imgs, 0), torch.Tensor(final_labels), torch.Tensor(dicom_ids)
            
            