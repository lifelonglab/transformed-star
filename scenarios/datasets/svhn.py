import copy
from typing import List, Tuple

import torchvision
from augly.image import RandomBlur, ShufflePixels, Opacity, Pixelization
from avalanche.benchmarks.datasets import SVHN
from torchvision.transforms import ToTensor, Resize, Compose, RandomPosterize, RandomInvert, RandomCrop

from cl_paths import DATA_PATH
from scenarios.utils import load_dataset


train_transform_with_resize = Compose([
    ToTensor(),
    Resize((32, 32)),
])

test_transform_with_resize = Compose([
    ToTensor(),
    Resize((32, 32))
])


def _load_svhn(split: str, transform):
    dataset = SVHN(root=f'{DATA_PATH}/data', split=split, download=True, transform=transform)
    dataset.targets = dataset.labels
    return dataset


def load_svhn_with_augmentation(augmentation_list, balanced=False) -> Tuple[List, List]:
    train_datasets = []
    test_datasets = []

    if not balanced:
        train = _load_svhn('train', train_transform_with_resize)
        test = _load_svhn('test', test_transform_with_resize)

        train_datasets.append(train)
        test_datasets.append(test)

        for augmentation_name, augmentation in augmentation_list.items():
            train_transform_with_aug = copy.deepcopy(train_transform_with_resize)
            train_transform_with_aug.transforms.insert(0, augmentation)

            test_transform_with_aug = copy.deepcopy(test_transform_with_resize)
            test_transform_with_aug.transforms.insert(0, augmentation)

            train = _load_svhn('train', train_transform_with_aug)
            test = _load_svhn('test', test_transform_with_aug)

            train_datasets.append(train)
            test_datasets.append(test)

    return train_datasets, test_datasets




def load_svhn_resized(balanced=False, number_of_samples_per_class=None):
    transform_func = Compose([ToTensor(), Resize((64, 64))])
    train = load_dataset(lambda transform: _load_svhn('train', transform), transform_func, balanced=balanced,
                         number_of_samples_per_class=number_of_samples_per_class)
    test = load_dataset(lambda transform: _load_svhn('test', transform), transform_func, balanced=balanced,
                        number_of_samples_per_class=number_of_samples_per_class)

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

    train, test = load_svhn_with_augmentation(aug, balanced=False)

    aug_normal = {'original' : None}

    aug = {**aug_normal, **aug}

    for dataset, aug_name in zip(train, aug):
        # for x in tr:
        #     if x[0].size() != (3, 28, 28):
        #         print(x[0].size())
        x = dataset[0]
        trans = torchvision.transforms.ToPILImage()
        out = trans(x[0])
        out.save(f'samples_new/svhn_{aug_name}.png')
        out.show()
