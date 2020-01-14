import numpy as np
from PIL import Image

import torchvision


class TransformTwice:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2


def get_imagenet(root, n_labeled,
                transform_normal=None, transform_aug=None, transform_val=None,
                download=True):
    base_dataset = torchvision.datasets.ImageNet(root, split='train', download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled / 10))

    train_labeled_dataset = ImageNet(root, train_labeled_idxs, split='train', transform=transform_normal)
    train_unlabeled_dataset = ImageNet_unlabeled(root, train_unlabeled_idxs, split='train',
                                                transform=TransformTwice(transform_aug, transform_normal))
    train_unlabeled_dataset2 = ImageNet_unlabeled(root, train_unlabeled_idxs, split='train',
                                                 transform=transform_val)

    val_dataset = ImageNet(root, val_idxs, split='train', transform=transform_val, download=True)
    test_dataset = ImageNet(root, split='test', transform=transform_val, download=True)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, train_unlabeled_dataset2, val_dataset, test_dataset


def train_val_split(labels, n_labeled_per_class,classes=1000):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

class ImageNet(torchvision.datasets.ImageNet):

    def __init__(self, root, indexs=None, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super(ImageNet, self).__init__(root, split=split,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.samples = self.samples[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class ImageNet_unlabeled(ImageNet):

    def __init__(self, root, indexs, split=True,
                 transform=None, target_transform=None,
                 download=False):
        super(ImageNet_unlabeled, self).__init__(root, indexs, split=split,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])


