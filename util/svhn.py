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


def get_svhn(root, n_labeled,val_size=-1,
                transform_1=None, transform_2=None, transform_val=None,
                download=True):
    base_dataset = torchvision.datasets.SVHN(root, split='train', download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.labels, n_labeled, val_size)

    train_labeled_dataset = SVHN_labeled(root, train_labeled_idxs, split='train', transform=transform_2)
    train_unlabeled_dataset = SVHN_unlabeled(root, train_unlabeled_idxs, split='train',
                                                transform=TransformTwice(transform_1, transform_2))
    train_unlabeled_dataset2 = SVHN_unlabeled(root, train_unlabeled_idxs, split='train',
                                                 transform=transform_val)
    if val_size>0:
        val_dataset = SVHN_labeled(root, val_idxs, split='train', transform=transform_val, download=True)
    else:
        val_dataset = None
    test_dataset = SVHN_labeled(root, split='test', transform=transform_val, download=True)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #val: {len(val_idxs)} #test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, train_unlabeled_dataset2, val_dataset, test_dataset


def train_val_split(labels, n_labeled,val_size,classes=10):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled//classes])
        if val_size>0:
            train_unlabeled_idxs.extend(idxs[n_labeled//classes:-val_size//classes])
            val_idxs.extend(idxs[-val_size//classes:])
        else:
            train_unlabeled_idxs.extend(idxs[n_labeled//classes:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

class SVHN_labeled(torchvision.datasets.SVHN):

    def __init__(self, root, indexs=None, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super(SVHN_labeled, self).__init__(root, split=split,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.labels)[indexs]
        else:
            self.targets = self.labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SVHN_unlabeled(SVHN_labeled):

    def __init__(self, root, indexs, split=True,
                 transform=None, target_transform=None,
                 download=False):
        super(SVHN_unlabeled, self).__init__(root, indexs, split=split,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])


