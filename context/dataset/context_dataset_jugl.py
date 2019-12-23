import os

import pandas as pd
import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils


class AgeGenderDataset(Dataset):
    """ Markable age/gender dataset """

    def __init__(self, csv_file, root_dir, transform=None):

        self.gender2Idx = {"male": 0, "female": 1, "other":-1}
        self.age2Idx = {"children": 0, "adult": 1, "toddler":2}

        self.root_dir = root_dir
        self.annot = pd.read_csv(os.path.join(self.root_dir, csv_file), index_col=0)
        self.transforms = transform


    def load(self, im_path):
        pil_image = Image.open(im_path).convert("RGB")
        # convert to BGR
        # image = np.array(pil_image)[:,:,[2,1,0]]
        return pil_image

    def __len__(self):
        return len(self.annot)


    def __getitem__(self, idx):

        annot = self.annot.iloc[idx]
        image_hash = annot.name
        gender = annot.gender
        age = annot.age
        human_bbox = [
            annot.x1,
            annot.y1,
            annot.x2,
            annot.y2
        ]

        image_path = os.path.join(self.root_dir,
                                image_hash +
                                  ".jpg"
                                )

        image = self.load(image_path)
        # crop human_bbox:
        image_crop = image.crop(human_bbox)
        if self.transforms:
            img = self.transforms(image_crop)

        ##DEBUG
        # import pdb;pdb.set_trace()
        #import matplotlib.pyplot as plt
        #_img = img.numpy().transpose(1,2,0)
        #plt.imsave("temp.jpg", _img)

        gender_target = torch.LongTensor([self.gender2Idx[gender]]).squeeze()
        age_target = torch.LongTensor([self.age2Idx[age]]).squeeze()

        return img, gender_target, age_target