import copy
from typing import Tuple, List

import torchvision
from augly.image import RandomBlur, ShufflePixels, Opacity, Pixelization
from avalanche.benchmarks.datasets import TinyImagenet
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomPosterize, RandomInvert, RandomCrop

from cl_paths import DATA_PATH
from scenarios.utils import filter_classes

train_transform_with_resize = Compose([
    ToTensor(),
    Resize((64, 64)),
    # Normalize(mean=(0.9221,), std=(0.2681,))
])

test_transform_with_resize = Compose([
    ToTensor(),
    Resize((64, 64)),
    # Normalize(mean=(0.9221,), std=(0.2681,))
])


def load_imagenet_with_augmentation(augmentation_list, balanced=False, num_classes=20) -> Tuple[
    List, List]:
    train_datasets = []
    test_datasets = []

    if not balanced:
        train = TinyImagenet(root=f'{DATA_PATH}/data', train=True, transform=train_transform_with_resize)
        test = TinyImagenet(root=f'{DATA_PATH}/data', train=False, transform=test_transform_with_resize)

        train_datasets.append(train)
        test_datasets.append(test)

        for augmentation_name, augmentation in augmentation_list.items():
            train_transform_with_aug = copy.deepcopy(train_transform_with_resize)
            train_transform_with_aug.transforms.insert(0, augmentation)

            test_transform_with_aug = copy.deepcopy(test_transform_with_resize)
            test_transform_with_aug.transforms.insert(0, augmentation)

            train = TinyImagenet(root=f'{DATA_PATH}/data/{augmentation_name}', download=True, train=True,
                                 transform=train_transform_with_aug)
            test = TinyImagenet(root=f'{DATA_PATH}/data/{augmentation_name}', download=True, train=False,
                                transform=test_transform_with_aug)

            train_datasets.append(train)
            test_datasets.append(test)

    datasets = [filter_classes(tr, te, list(range(num_classes))) for tr, te in zip(train_datasets, test_datasets)]

    return [t for t, _ in datasets], [t for _, t in datasets]


if __name__ == '__main__':
    # aug = {
    #     'RandomBlur': RandomBlur(min_radius=2.0, max_radius=8.0),
    #     'RandomPosterize':  RandomPosterize(bits=2, p=1.0),
    #     'RandomInvert':  RandomInvert(p=1.0),
    #     'RandomCrop': RandomCrop(size=20, pad_if_needed=True),
    # }

    # aug = {
    #     'RandomPerspective': RandomPerspective()
    # }

    aug = {'ShufflePixels': ShufflePixels(factor=0.5),
           'Opacity': Opacity(level=0.5),
           'Pixelization': Pixelization(ratio=0.75)
           }

    train, test = load_imagenet_with_augmentation(aug, balanced=False)

    aug_normal = {'original': None}

    aug = {**aug_normal, **aug}

    for dataset, aug_name in zip(train, aug):
        # for x in tr:
        #     if x[0].size() != (3, 28, 28):
        #         print(x[0].size())
        x = dataset[0]
        trans = torchvision.transforms.ToPILImage()
        out = trans(x[0])
        out.save(f'samples_new/tiny_{aug_name}.png')
        out.show()
