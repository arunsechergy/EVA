import albumentations as A
from albumentations.pytorch import ToTensorV2


class MNISTTransforms:
    def __init__(self):
        pass

    def build_transforms(self, train_tfms_list=[], test_tfms_list=[]):
        train_tfms_list.extend([A.Normalize(mean=[0.1307], std=[0.3081]), ToTensorV2()])
        test_tfms_list.extend([A.Normalize(mean=[0.1307], std=[0.3081]), ToTensorV2()])
        return A.Compose(train_tfms_list), A.Compose(test_tfms_list)

class CIFAR10Transforms:
    def __init__(self):
        pass

    def build_transforms(self,  train_tfms_list=[], test_tfms_list=[]):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_tfms_list.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
        test_tfms_list.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
        return A.Compose(train_tfms_list), A.Compose(test_tfms_list)
