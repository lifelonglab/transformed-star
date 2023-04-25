from typing import List, Tuple

import torchvision
from augly.image import RandomBlur, ShufflePixels, Opacity, Pixelization
from avalanche.benchmarks.datasets import CIFAR10
from matplotlib import pyplot as plt
from torchvision.transforms import Resize, Compose, ToTensor, RandomCrop, Normalize, RandomPosterize, RandomInvert

import copy

from cl_paths import DATA_PATH
from scenarios.utils import load_dataset

train_transform = Compose([
    RandomCrop(32, padding=4),
    ToTensor(),
    Normalize(mean=(0.9221,), std=(0.2681,))
])

test_transform = Compose([
    ToTensor(),
    # Normalize(mean=(0.9221,), std=(0.2681,))
])

train_transform_with_resize = Compose([
    ToTensor(),
    Resize((32, 32)),
    # Normalize(mean=(0.9221,), std=(0.2681,))
])

test_transform_with_resize = Compose([
    ToTensor(),
    Resize((32, 32)),
    # Normalize(mean=(0.9221,), std=(0.2681,))
])


def load_cifar10():
    train = CIFAR10(root=f'{DATA_PATH}/data', download=True, train=True, transform=train_transform_with_resize)
    test = CIFAR10(root=f'{DATA_PATH}/data', download=True, train=False, transform=test_transform_with_resize)

    return train, test


def load_resized_cifar10(balanced=False, number_of_samples_per_class=None):
    train = load_dataset(
        lambda transform: CIFAR10(root=f'{DATA_PATH}/data', download=True, train=True, transform=transform),
        train_transform_with_resize, balanced=balanced,
        number_of_samples_per_class=number_of_samples_per_class)
    test = load_dataset(
        lambda transform: CIFAR10(root=f'{DATA_PATH}/data', download=True, train=False, transform=transform),
        test_transform_with_resize, balanced=balanced,
        number_of_samples_per_class=number_of_samples_per_class)
    return train, test


def load_cifar10_with_augmentation(augmentation_list, balanced=True) -> Tuple[
    List, List]:
    train_datasets = []
    test_datasets = []
    if not balanced:
        train = CIFAR10(root=f'{DATA_PATH}/data/', download=True, train=True, transform=train_transform_with_resize)
        test = CIFAR10(root=f'{DATA_PATH}/data/', download=True, train=False, transform=test_transform_with_resize)
        train_datasets.append(train)
        test_datasets.append(test)
        for augmentation_name, augmentation in augmentation_list.items():
            train_transform_with_aug = copy.deepcopy(train_transform_with_resize)
            train_transform_with_aug.transforms.insert(0, augmentation)

            test_transform_with_aug = copy.deepcopy(test_transform_with_resize)
            test_transform_with_aug.transforms.insert(0, augmentation)

            train = CIFAR10(root=f'{DATA_PATH}/data/{augmentation_name}', download=True, train=True,
                            transform=train_transform_with_aug)
            test = CIFAR10(root=f'{DATA_PATH}/data/{augmentation_name}', download=True, train=False,
                           transform=test_transform_with_aug)

            train_datasets.append(train)
            test_datasets.append(test)

        return train_datasets, test_datasets



if __name__ == '__main__':
    # aug = {
    #     'RandomBlur': RandomBlur(min_radius=2.0, max_radius=8.0),
    #     'RandomPosterize':  RandomPosterize(bits=2, p=1.0),
    #     'RandomInvert':  RandomInvert(p=1.0),
    #     'RandomCrop': RandomCrop(size=20, pad_if_needed=True),
    # }

    aug = {'ShufflePixels': ShufflePixels(factor=0.5),
           'Opacity': Opacity(level=0.5),
           'Pixelization': Pixelization(ratio=0.75)
           }

    # aug = {
    #     'RandomPerspective': RandomPerspective()
    # }

    train, test = load_cifar10_with_augmentation(aug, balanced=False)

    aug_normal = {'original' : None}

    aug = {**aug_normal, **aug}

    for dataset, aug_name in zip(train, aug):
        # for x in tr:
        #     if x[0].size() != (3, 28, 28):
        #         print(x[0].size())
        x = dataset[0]
        trans = torchvision.transforms.ToPILImage()
        out = trans(x[0])
        out.save(f'samples_new/cifar_{aug_name}.png')
        out.show()

        # plt.imshow(out)