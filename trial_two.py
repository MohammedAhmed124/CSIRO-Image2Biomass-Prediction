import os
import cv2
import pandas as pd

import torch
import torch.nn as nn


import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import glob
from collections import defaultdict


from torch.cuda.amp import GradScaler

import matplotlib
matplotlib.use("Agg")
from utils.config import CFG
from utils.training import run_kfold_training
from torch.utils.data import Dataset


def get_id(path):
    return os.path.splitext(os.path.basename(path))[0]

base_directory = CFG.DATA_DIR
train_csv = pd.read_csv(
    os.path.join(base_directory , "train.csv")
)
train_imgs_path = glob.glob(\
    os.path.join(base_directory , "train" , "*.jpg")
)

dict_ = defaultdict(list)

for path in train_imgs_path:
    id_ = get_id(path)
    cols_to_include = ["target_name" , "target" , "Sampling_Date" , "State" ,"Pre_GSHH_NDVI" ]
    img_info = train_csv[train_csv["sample_id"].str.split("_" , expand=True)[0]==id_][cols_to_include].copy()
    extra_info = img_info[[ "Sampling_Date" , "State" ,"Pre_GSHH_NDVI"]].iloc[0,:].to_dict()
    info_dict = img_info.set_index("target_name").to_dict()['target']
    info_dict["id"] = id_
    info_dict["image_path"] = path
    info_dict.update(extra_info)
    for k  , v in info_dict.items():
        dict_[k].append(v)

df = pd.DataFrame(dict_)







class BiomassDataset(Dataset):
    """
    Custom PyTorch Dataset for the CSIRO biomass competition.
    It reads the 2000x1000 image and splits it into two halves (left and right).
    """
    def __init__(
            self,
            df,
            image_dir,
            transforms,
            targets
            ):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.targets = targets
        
        # Get image paths and labels from the dataframe
        self.image_paths = self.df['image_path'].values
        self.labels = self.df[self.targets].values
        # print(self.image_paths)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get image path and labels
        # img_path = os.path.join(self.image_dir, self.image_paths[idx])
        img_path = self.image_paths[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        # 2. Read the image
        # [cite_start]Images are 2000px wide, 1000px high [cite: 181]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. **C Split image into left and right halves**
        # The original image is 2000x1000. We split it at the 1000px midpoint.
        width = img.shape[1]
        mid_point = width // 2

        
        img_left = img[:, :mid_point]  # First half (1000x1000)
        img_right = img[:, mid_point:] # Second half (1000x1000)

        # 4. Apply augmentations to *both* halves

        if self.transforms:
            img_left = self.transforms(image=img_left)['image']
            img_right = self.transforms(image=img_right)['image']
        return {
            "left_image": img_left,
            "right_image": img_right,
            "labels": labels
        }

class BiomassModel(nn.Module):
    """
    A "Two-Stream" model. It processes the left and right images separately,
    combines their features, and then predicts the 3 main targets.
    """
    def __init__(
            self,
            model_name,
            create_model_params = None,
            dropout_rate = 0.2,
            ):
        if create_model_params is None:
            create_model_params = dict(
            pretrained=False,
            num_classes=0,
            global_pool='avg'
            )

        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            **create_model_params
        )

        n_features = self.backbone.num_features
        

        self.meta_mlp = None
        n_fused = n_features * 2

                
        # Three prediction heads: Total, GDM, Green
        self.head_total = self._create_head(n_fused, dropout_rate)
        self.head_gdm = self._create_head(n_fused, dropout_rate)
        self.head_green = self._create_head(n_fused, dropout_rate)
    
    def _create_head(self, in_features: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, 1)
        )
        

    def forward(self, left_image, right_image):
        # 1. Get features for left image
        feat_left = self.backbone(left_image)
        
        # 2. Get features for right image
        feat_right = self.backbone(right_image)
        
        # 3. Combine features
        # (batch_size, n_features * 2)
        features = torch.cat([feat_left, feat_right], dim=1)
        
        # 4. Get predictions from each head
        pred_total = self.head_total(features)
        pred_gdm = self.head_gdm(features)
        pred_green = self.head_green(features)
        
        return pred_total, pred_gdm, pred_green



# model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True)
# data_config = timm.data.resolve_model_data_config(model)
# train_tfm = timm.data.create_transform(**data_config, is_training=True)
# val_tfm   = timm.data.create_transform(**data_config, is_training=False)


def get_transforms(img_size):
    """
    Returns simple augmentations for training and validation.
    """
    # For training
    train_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #These are the global mean and standard deviation of the ImageNet dataset, after scaling pixels to [0, 1].
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(img_size, img_size),
        #These are the global mean and standard deviation of the ImageNet dataset, after scaling pixels to [0, 1].
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transforms, val_transforms
# define model_constructor: function that returns a model instance
def model_constructor(constructor_params):
    if constructor_params is None:
        constructor_params = {}
    return BiomassModel(
        **constructor_params
        )


model_kwargs = dict(constructor_params = dict(
    model_name = "eva02_base_patch14_448.mim_in22k_ft_in1k",
    dropout_rate = 0.2,
    create_model_params = dict(
            pretrained=True,
            num_classes=0,
            global_pool='avg'
            )
)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_config = {
    "batch_size": 4,
    "accumulation_steps":6,                    
    "n_epochs": 80,                        
    "optimizer": torch.optim.AdamW,       
    "optimizer_config": {
        "lr": 1e-4,                        
        "weight_decay": 1e-2,              
    },
    "criterion": nn.MSELoss,              
    "accumulation_steps": 4,               
    "scheduler": None,                     
    "scheduler_config": {},                
    "seed": 42,                            
    "scaler": GradScaler,
    "scaler_config":{
        "enabled":(device.type == "cuda")
        }                       
}

out= run_kfold_training(    
        model_constructor=model_constructor,
        df=df,              # pandas DataFrame with all training rows
        training_config=training_config,
        BiomassDatasetClass=BiomassDataset,
        get_transforms=get_transforms, # callable(img_size) -> (train_tfm, val_tfm)
        model_kwargs=model_kwargs,
        kfold_params = {
            "strategy":"stratified",
            "random_state":42,
            # "group_col":"Sampling_Date",
            "stratify_col":"State",
        },
        transforms_config={
            "img_size":448
        },


        ds_kwargs={}, 
        n_splits=4,
        output_dir = "runs/EVA02",
        device= device,
        save_every= 5,
        num_workers = 0,
        seed = 42,
        continue_=False
        )
