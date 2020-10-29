from dataset.autoaugment import RandAugment
import torchvision.transforms as transforms
def get_data_augment(dataset):
    if dataset == 'cifar10':
        means = (0.4914, 0.4822, 0.4465)
        stds = (0.2471, 0.2435, 0.2616)
        transform_aug = transforms.Compose([
             RandAugment(3, 5),
             transforms.RandomCrop(32, padding=4),  # fill parameter needs torchvision installed from source
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(means, stds),
             ])
        transform_normal = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

    if dataset == 'svhn':
        means = (0.4376821, 0.4437697, 0.47280442)
        stds = (0.19803012, 0.20101562, 0.19703614)
        transform_aug = transforms.Compose([
            RandAugment(3, 7),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        transform_normal = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        #
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    return transform_aug, transform_normal, transform_val