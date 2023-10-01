from .voc_dataset import VOCDataset
from .coco_dataset import COCODataset
from .mov_dataset import MOVDataset
from .imagenetvid_dataset import ImagenetVIDDataset

__all__ = ["datasets", "collate_wrapper"]


def datasets(ds, *args, **kwargs):
    ds = ds.lower()
    choice = ["voc", "coco", "movmnist", "imagenetvid"]
    if ds == choice[0]:
        return VOCDataset(*args, **kwargs)
    if ds == choice[1]:
        return COCODataset(*args, **kwargs)
    if ds ==choice[2]:
        return MOVDataset(*args, **kwargs)
    if ds == choice[3]:
        return ImagenetVIDDataset(*args, **kwargs)
    else:
        raise ValueError("'ds' must be in '{}', but got '{}'".format(choice, ds))
    
    
def collate_wrapper(batch):
    return CustomBatch(batch)

    
class CustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.images = transposed_data[0]
        self.targets = transposed_data[1]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.images = [img.pin_memory() for img in self.images]
        self.targets = [{k: v.pin_memory() for k, v in tgt.items()} for tgt in self.targets]
        return self
    
