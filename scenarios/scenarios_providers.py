from augly.image import RandomBlur
from avalanche.benchmarks import dataset_benchmark
from torchvision.transforms import RandomCrop, RandomPosterize, RandomInvert

from scenarios.datasets.cifar import load_cifar10_with_augmentation
from scenarios.datasets.cub import load_cube200_with_augmentation
from scenarios.datasets.imagnet import load_imagenet_with_augmentation
from scenarios.datasets.mnist import load_mnist_with_augmentation
from scenarios.datasets.svhn import load_svhn_with_augmentation


def _get_cl_augs():
    return {
        'RandomBlur': RandomBlur(min_radius=4.0, max_radius=8.0),
        'RandomPosterize': RandomPosterize(bits=4, p=1.0),
        'RandomInvert': RandomInvert(p=1.0),
        'RandomCrop': RandomCrop(size=30, pad_if_needed=True),
    }


def get_simple_mnist():
    train, test = load_mnist_with_augmentation(_get_cl_augs(), balanced=False)

    scenario = dataset_benchmark(train_datasets=train, test_datasets=test)

    return scenario


def get_cifar10_scenario():
    train_cifar10_datasets, test_cifar10_datasets = load_cifar10_with_augmentation(_get_cl_augs(), balanced=False)

    scenario = dataset_benchmark(train_datasets=train_cifar10_datasets, test_datasets=test_cifar10_datasets)

    return scenario


def get_imagenet_scenario(num_classes=200):
    train_imagenet_datasets, test_imagenet_datasets = load_imagenet_with_augmentation(_get_cl_augs(), balanced=False, num_classes=num_classes)

    scenario = dataset_benchmark(train_datasets=train_imagenet_datasets, test_datasets=test_imagenet_datasets)

    return scenario


def get_svhn_scenario():
    train_svhn_datasets, test_svhn_datasets = load_svhn_with_augmentation(_get_cl_augs(), balanced=False)

    scenario = dataset_benchmark(train_datasets=train_svhn_datasets, test_datasets=test_svhn_datasets)

    return scenario


def get_cube200_scenario():
    train_cube200_datasets, test_cube200_datasets = load_cube200_with_augmentation(_get_cl_augs(), balanced=False)

    scenario = dataset_benchmark(train_datasets=train_cube200_datasets, test_datasets=test_cube200_datasets)

    return scenario


def parse_scenario(args):
    if args.scenario == 'simple_mnist':
        return get_simple_mnist()
    elif args.scenario == 'cifar10':
        return get_cifar10_scenario()
    elif args.scenario == 'imagenet':
        return get_imagenet_scenario(args.num_classes)
    elif args.scenario == 'svhn':
        return get_svhn_scenario()
    elif args.scenario == 'cube200':
        return get_cube200_scenario()
    else:
        raise NotImplementedError("SCENARIO NOT IMPLEMENTED YET: ", args.scenario)


if __name__ == '__main__':
    scenario = get_simple_mnist()
    for t in scenario.train_stream:
        print(t)
