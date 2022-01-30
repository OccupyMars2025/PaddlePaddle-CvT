import os

from paddle.vision import datasets, transforms
# 构建dataset
from paddle.io import Dataset, DataLoader
import cv2
import os

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ImageNetDataset(Dataset):
    """

    """
    def __init__(self, data_dir, info_txt, transforms=None):
        super(ImageNetDataset, self).__init__()
        self.data_dir = data_dir
        self.image_paths, self.labels = self.get_info(info_txt)
        self.transforms = transforms

    def get_info(self, file_path):
        paths = []
        labels = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_name, label = line.strip().split(' ')
                paths.append(os.path.join(self.data_dir, image_name))
                labels.append(int(label))
        return paths, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        if self.transforms:
            image = self.transforms(image)
        return image, label


def transforms_imagenet_train(img_size=224):
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(img_size),
         transforms.RandomHorizontalFlip(prob=0.5),
         transforms.RandomVerticalFlip(prob=0.2),
         transforms.ToTensor(),
         transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
         ]
    )
    return transform


def transforms_imagenet_eval(img_size=224):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    return transform


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    dataset, nb_classes = None, None
    if args.data_set == 'CIFAR':
        dataset = datasets.Cifar100(data_file=args.data_path, mode='train' if is_train else 'test',
                                    transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')

        dataset = ImageNetDataset(data_dir=args.data_path,
                                  info_txt=args.train_info_txt if is_train else args.val_info_txt,
                                  transforms=transform)
        nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, args):
    if is_train:
        return transforms_imagenet_train()
    else:
        return transforms_imagenet_eval()
