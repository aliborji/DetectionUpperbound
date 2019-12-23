import os

import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import json

class ContextDatasetNew(Dataset):
    """ Markable age/gender dataset """

    def __init__(self, root_dir, json_file, img_dir, transform=None, context_alpha = 1, context_only = False): #csv_file, root_dir, transform=None):

        # self.gender2Idx = {"male": 0, "female": 1, "other":-1}
        # self.age2Idx = {"children": 0, "adult": 1, "toddler":2}  # convet to object!!
        # self.age2Idx = {"children": 0, "adult": 1, "toddler":2}  # convet to object!!        
        # import pdb; pdb.set_trace()
        self.root_dir = root_dir
        self.img_dir = img_dir

        if 'coco' in root_dir:
           if 'val' in json_file:
              self.img_dir += 'val2017'
           else:     
              self.img_dir += 'train2017'    

        if 'openImages' in root_dir:
           if 'val' in json_file:
              self.img_dir += 'validation'
           else:
              self.img_dir += 'train'    

        
        # self.annot = pd.read_csv(os.path.join(self.root_dir, csv_file), index_col=0) 
        # borji; read from json instead of csv
        with open(os.path.join(self.root_dir, json_file)) as json_file:
            data = json.load(json_file)        

        # create a list where each item is a dictionart including bbox + img_path + obj_cat    

        # category ids for pascal and markable start from 1




        mapp = {}
        ws = {}
        hs = {}
        for k in data['images']:
            mapp[k['id']] = k['file_name']
            ws[k['id']]   = k['width']
            hs[k['id']]   = k['height']

        self.annot = []

        if ('markable' in self.root_dir) or ('voc' in self.root_dir):
            # inverted_dict = dict([[v,k] for k,v in mapp.items()])
            for i in data['annotations']:
               self.annot.append({'box':i['bbox'], 'segmentation':i['segmentation'], 'category_id':i['category_id'], 'file_name': mapp[i['image_id']], 'im_w': ws[i['image_id']], 'im_h': hs[i['image_id']]})

        if 'coco' in self.root_dir:    
            valid_ids = [
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
              24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
              37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
              58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
              72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
              82, 84, 85, 86, 87, 88, 89, 90]
            cat_ids = {v: i for i, v in enumerate(valid_ids, 1)}
           
            # inverted_dict = dict([[v,k] for k,v in mapp.items()])
            for i in data['annotations']:
               self.annot.append({'box':i['bbox'], 'category_id':cat_ids[i['category_id']], 'file_name': mapp[i['image_id']], 'im_w': ws[i['image_id']], 'im_h': hs[i['image_id']]})

        if 'openImages' in self.root_dir:  # category starts from 0          
            print('hey openImages!!!')
            for i in data['annotations']:  
               self.annot.append({'box':i['bbox'],  'category_id':i['category_id']+1, 'file_name': mapp[i['image_id']], 'im_w': ws[i['image_id']], 'im_h': hs[i['image_id']]})











        # import pdb; pdb.set_trace()   
        self.transforms = transform
        self.context_alpha = context_alpha
        self.context_only = context_only      


    def load_old(self, im_path):
            # pil_image = Image.open(im_path).convert("RGB")
            # # convert to BGR
            # # image = np.array(pil_image)[:,:,[2,1,0]]
            # return pil_image

            cv2_image = cv2.imread(im_path)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            # convert to BGR
            # image = np.array(pil_image)[:,:,[2,1,0]]
            return cv2_image




    def load(self, im_path):
        pil_image = Image.open(im_path).convert("RGB")
        # convert to BGR
        # image = np.array(pil_image)[:,:,[2,1,0]]
        return pil_image




    def __len__(self):
        return len(self.annot)




    def __getitem__(self, idx):

        # annot = self.annot.iloc[idx]
        # import pdb; pdb.set_trace()
        annot = self.annot[idx]        
        image_hash = annot['file_name'] #annot.name
        obj_cat = annot['category_id'] - 1 # borji; to bring the range from [1 n] to [0 n-1]
        

        obj_bbox = annot['box']
        obj_bbox[2] += obj_bbox[0] # x + w
        obj_bbox[3] += obj_bbox[1] # y + w       




        image_path = os.path.join(self.img_dir,  # 
                                image_hash ) #+
                                #   ".jpg"
                                # )

        image = self.load(image_path)

        image_crop = image.crop(obj_bbox)



        if self.transforms:
            img = self.transforms(image_crop)


        ##DEBUG
        # import pdb;pdb.set_trace()
        #import matplotlib.pyplot as plt
        #_img = img.numpy().transpose(1,2,0)
        #plt.imsave("temp.jpg", _img)

        # gender_target = torch.LongTensor([self.gender2Idx[gender]]).squeeze()
        # age_target = torch.LongTensor([self.age2Idx[age]]).squeeze()
        # obj_target = torch.LongTensor(obj_cat) #.squeeze()        
        obj_target = obj_cat #.squeeze()        

        # return img, gender_target, age_target
        return img, obj_target        