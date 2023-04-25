import copy
from typing import List, Tuple

import torchvision
from augly.image import RandomBlur, RandomNoise, Scale, EncodingQuality, Grayscale, RandomBrightness, ShufflePixels, \
    Opacity, Pixelization
from avalanche.benchmarks.datasets import MNIST
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, Resize, RandomPosterize, RandomInvert

from cl_paths import DATA_PATH
from scenarios.utils import transform_from_gray_to_rgb, balance_dataset, load_dataset

train_transform = Compose([
    RandomCrop(28, padding=4),
    ToTensor(),
    transform_from_gray_to_rgb(),
    # Normalize((0.1307,), (0.3081,))
])

test_transform = Compose([
    ToTensor(),
    transform_from_gray_to_rgb(),
    Normalize((0.1307,), (0.3081,))
])

train_transform_with_resize = Compose([
    ToTensor(),
    Resize((64, 64)),
    transform_from_gray_to_rgb(),
    # Normalize((0.1307,), (0.3081,))
])

test_transform_with_resize = Compose([
    ToTensor(),
    Resize((64, 64)),
    transform_from_gray_to_rgb(),
    Normalize((0.1307,), (0.3081,))
])


def load_mnist(balanced=True, number_of_samples_per_class=None):
    if not balanced:
        train = MNIST(root=f'{DATA_PATH}/data', download=True, train=True, transform=train_transform)
        test = MNIST(root=f'{DATA_PATH}/data', download=True, train=False, transform=test_transform)

        return train, test

    train = MNIST(root=f'{DATA_PATH}/data', download=True, train=True)
    test = MNIST(root=f'{DATA_PATH}/data', download=True, train=False)

    train = balance_dataset(train, train_transform, number_of_samples_per_class)
    test = balance_dataset(test, test_transform, number_of_samples_per_class)

    return train, test


def load_mnist_with_augmentation(augmentation_list, balanced=True, number_of_samples_per_class=None) -> Tuple[
    List, List]:
    train_datasets = []
    test_datasets = []
    if not balanced:
        train = MNIST(root=f'{DATA_PATH}/data/', download=True, train=True, transform=train_transform)
        test = MNIST(root=f'{DATA_PATH}/data/', download=True, train=False, transform=test_transform)
        train_datasets.append(train)
        test_datasets.append(test)
        for augmentation_name, augmentation in augmentation_list.items():
            train_transform_with_aug = copy.deepcopy(train_transform)
            train_transform_with_aug.transforms.insert(0, augmentation)

            test_transform_with_aug = copy.deepcopy(test_transform)
            test_transform_with_aug.transforms.insert(0, augmentation)

            train = MNIST(root=f'{DATA_PATH}/data/{augmentation_name}', download=True, train=True,
                          transform=train_transform_with_aug)
            test = MNIST(root=f'{DATA_PATH}/data/{augmentation_name}', download=True, train=False,
                         transform=test_transform_with_aug)

            train_datasets.append(train)
            test_datasets.append(test)

        return train_datasets, test_datasets

    #     train = MNIST(root=f'{DATA_PATH}/data', download=True, train=True)
    #     test = MNIST(root=f'{DATA_PATH}/data', download=True, train=False)
    #
    #     train = balance_dataset(train, train_transform, number_of_samples_per_class)
    #     test = balance_dataset(test, test_transform, number_of_samples_per_class)
    #
    # return train, test


def load_mnist_with_resize(balanced=False, number_of_samples_per_class=None):
    train = load_dataset(
        lambda transform: MNIST(root=f'{DATA_PATH}/data', download=True, train=True, transform=transform),
        balanced=balanced, number_of_samples_per_class=number_of_samples_per_class,
        transform=train_transform_with_resize)

    test = load_dataset(
        lambda transform: MNIST(root=f'{DATA_PATH}/data', download=True, train=False, transform=transform),
        balanced=balanced, number_of_samples_per_class=number_of_samples_per_class,
        transform=test_transform_with_resize)

    return train, test


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

    train, test = load_mnist_with_augmentation(aug, balanced=False)

    aug_normal = {'original' : None}

    aug = {**aug_normal, **aug}

    for dataset, aug_name in zip(train, aug):
        # for x in tr:
        #     if x[0].size() != (3, 28, 28):
        #         print(x[0].size())
        x = dataset[0]
        trans = torchvision.transforms.ToPILImage()
        out = trans(x[0])
        out.save(f'samples_new/mist{aug_name}.png')
        out.show()