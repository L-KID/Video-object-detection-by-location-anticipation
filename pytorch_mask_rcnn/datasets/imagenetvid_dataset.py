import os
from PIL import Image

import torch
from .generalized_dataset import GeneralizedDataset
   

# CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
#                'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
#                'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
#                'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
#                'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
#                'zebra')

# CLASSES = ('dog', 'giant_panda', 'hamster')
CLASSES = ('dog', 'giant_panda', 'hamster')

        
class ImagenetVIDDataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        ann_file = os.path.join(data_dir, "annotations/imagenet_vid_{}.json".format(split))
        self.coco = COCO(ann_file)
        tmp = [str(k) for k in self.coco.imgs]
        self.ids = []
        count = 0
        prev_video_ID = ''
        for ID in tmp:
            tar = self.get_tem_target(ID)
            video_ID = tar['video_id']
            if len(tar['labels']) <= 1 and (tar['labels'] == 13 or tar['labels'] == 9 or tar['labels'] == 14):
                if prev_video_ID == '':
                    prev_video_ID = video_ID
                    count = 1
                    self.ids.append(ID)
                elif video_ID == prev_video_ID and count < 60:
                    self.ids.append(ID)
                    count += 1
                    # print(ID)
                elif video_ID != prev_video_ID:
                    while count < 60:
                        self.ids.append(self.ids[-1])
                        count += 1
                    count = 1
                    prev_video_ID = video_ID
                    self.ids.append(ID)
                    # print(ID)
        # for ID in self.ids:
        #     tar = self.get_target(ID)
        #     print(tar)

        # classes's values must start from 1, because 0 means background in the model
        # self.classes = {k: v["name"] for k, v in self.coco.cats.items()}
        self.classes = {i: n for i, n in enumerate(CLASSES, 1)}
        
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            self.check_dataset(checked_id_file)
            print('length', len(self.ids))
        
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join('../ILSVRC2015/Data/VID', img_info["file_name"]))
        return image.convert("RGB")
    
    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []
        video_id = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                video_id.append(ann['video_id'])
                # mask = self.coco.annToMask(ann)
                # mask = torch.tensor(mask, dtype=torch.uint8)
                # masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            # print('boxes', boxes)
            boxes = self.convert_to_xyxy(boxes)
            for i, item in enumerate(labels):
                if item == 9: labels[i] = 1
                elif item == 13: labels[i] = 2
                elif item == 14: labels[i] = 3

            labels = torch.tensor(labels)
            video_id = torch.tensor(video_id)
            # print('labels', labels)
            # masks = torch.stack(masks)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, video_id=video_id, masks=None)
        # print('target', target)
        return target

    def get_tem_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []
        video_id = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                video_id.append(ann['video_id'])
                # mask = self.coco.annToMask(ann)
                # mask = torch.tensor(mask, dtype=torch.uint8)
                # masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            # print('boxes', boxes)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            video_id = torch.tensor(video_id)
            # print('labels', labels)
            # masks = torch.stack(masks)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, video_id=video_id, masks=None)
        # print('target', target)
        return target
    
    
