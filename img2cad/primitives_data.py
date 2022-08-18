import dataclasses
import math
import logging
import os

from typing import List, Optional, Tuple, Union

import numpy as np
import omegaconf
import pytorch_lightning
import torch
import torch.utils.data

from img2cad import dataset, modules


@dataclasses.dataclass
class PrimitiveDataConfig:
    """Configuration for creating a new `PrimitiveDataset`.

    Attributes
    ----------
    num_position_bins
        Number of bins used for quantization of position.
    sequence_path
        Path to file containing sequence data.
    max_token_length
        Maximum length for token sequences representing primitives.
    validation_fraction : float
        Fraction of data to use as validation set.
    test_fraction : float
        Fraction of data to use as test set.
    """
    num_position_bins: int = 64
    sequence_path: str = 'data/sg_filtered_unique.npy'
    max_token_length: int = 130
    dataset_size: Optional[int] = None
    validation_fraction: Optional[float] = 0.025
    test_fraction: Optional[float] = 0.05


@dataclasses.dataclass
class RawPrimitiveDataConfig(PrimitiveDataConfig):
    """Configuration for creating a `PrimitiveDataset` for creating raw primitive models.

    Attributes
    ----------
    permute_entities : bool
        If `True`, indicates that entity order should be permuted in the primitive dataset.
    """
    permute_entities: bool = False


@dataclasses.dataclass
class ImageAugmentationConfig:
    """Configuration for specifying image transformations.

    Attributes
    ----------
    enabled : bool
        Whether image transformations are enabled.
    shift_fraction : float
        Fraction of the image by which to randomly translate the image.
    rotation : float
        Maximal random rotation amount in degrees.
    shear : float
        Maximal random shear amount in degrees.
    scale : float
        Maximum proportion by which to rescale the image by.
    """
    enabled: bool = True
    shift_fraction: float = 12 / 128
    rotation: float = 8
    shear: float = 8
    scale: float = 0.2


@dataclasses.dataclass
class ImagePrimitiveDataConfig(PrimitiveDataConfig):
    """Configuration for creating a new `ImagePrimitiveDataModule`.

    Attributes
    ----------
    image_data_folder : str, optional
        Path to the folder containing image sequence data. If `None`, this is inferred
        from the sequence file location.
    augment : ImageAugmentation
        Configuration for the image augmentation to apply.
    use_noisy_images : bool
        If `True`, indicates that noisy renderings from the image should be used.
        Otherwise, uses reference renderings.
    """
    image_data_folder: Optional[str] = None
    augmentation: ImageAugmentationConfig = ImageAugmentationConfig()
    use_noisy_images: bool = True


def _get_cache_data_path(sequence_path: os.PathLike, num_position_bins: int) -> str:
    path, _ = os.path.splitext(sequence_path)
    return path + f'_b{num_position_bins}.cache.npz'


def get_image_folder_from_sequence(sequence_path: os.PathLike) -> str:
    """Computes the location of the image folder from the location of the sequence file."""
    return os.path.join(os.path.dirname(sequence_path), 'renders')


def cache_data_if_needed(sequence_path: os.PathLike, num_position_bins: int) -> bool:
    """Generates cached sequence data if does not exist.

    Parameters
    ----------
    sequence_path : os.PathLike
        Path to the file containing raw sequences.
    num_position_bins : int
        Number of bins to use for quantiziing positional parameters.
    """
    cache_data_path = _get_cache_data_path(sequence_path, num_position_bins)

    log = logging.getLogger(__name__)
    if os.path.exists(cache_data_path):
        log.info('Found cached data at location {}'.format(cache_data_path))
        return False

    log.info('Cached data not found. Generating cache.')
    primitive_dataset = dataset.PrimitiveDataset(sequence_path, num_position_bins)

    data = dataset.process_sequence_data(primitive_dataset)
    data = {k: np.asarray(v) for k, v in data.items()}

    log.info('Done processing data, saving data to {}'.format(cache_data_path))
    np.savez(cache_data_path, **data)
    return True


def split_dataset(dataset: torch.utils.data.Dataset,
                  validation_fraction: Optional[float]=None,
                  test_fraction: Optional[float]=None) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """Splits the given dataset into a training dataset and a validation dataset.
    The split is performed randomly (but the same every time).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to split.
    validation_fraction : float, optional
        If not `None`, the fraction of samples to reserve for the validation set.

    Returns
    -------
    train_dataset : torch.utils.data.Dataset
        The training dataset split
    valid_dataset : Optional[torch.utils.data.Dataset]
        If ``validation_fraction`` is not `None` and greater than 0,
        the validation dataset split.
    """

    generator = torch.Generator().manual_seed(4242424242)
    perm = torch.randperm(len(dataset), generator=generator)

    if validation_fraction is None:
        validation_fraction = 0.0

    if test_fraction is None:
        test_fraction = 0.0

    num_train_samples = int(math.ceil(len(dataset) * (1 - validation_fraction - test_fraction)))
    num_test_samples = int(math.ceil(len(dataset) * test_fraction))

    if num_train_samples + num_test_samples > len(dataset):
        num_test_samples = len(dataset) - num_train_samples

    num_valid_samples = len(dataset) - num_train_samples - num_test_samples

    train_idx, valid_idx, test_idx = torch.split(perm, [num_train_samples, num_valid_samples, num_test_samples])

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, valid_dataset, test_dataset


class PrimitiveDataModule(pytorch_lightning.LightningDataModule):
    """This class provides a wrapper over `dataset.PrimitiveDataset` to handle data loading.
    """
    def __init__(self, config: RawPrimitiveDataConfig, batch_size: int=128, num_workers: int=8):
        super().__init__()

        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._dataset = None
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._valid_dataset_no_noise = None
        self._test_dataset_no_noise = None

    def prepare_data(self) -> None:
        cache_data_if_needed(self.config.sequence_path, self.config.num_position_bins)

    def setup(self):
        dataset_args = {
            'sequence_file': self.config.sequence_path,
            'num_bins': self.config.num_position_bins,
            'max_length': self.config.max_token_length
        }

        if getattr(self.config, 'permute_entities', False):
            # Use getattr to preserve compatibility with some saved models
            logging.info('Loading permuted primitive dataset.')
            self._dataset = dataset.PrimitiveDataset(**dataset_args, permute=True)
            non_noise_dataset = dataset.PrimitiveDataset(**dataset_args, permute=False)

            _, self._valid_dataset_no_noise, self._test_dataset_no_noise = split_dataset(
                non_noise_dataset, self.config.validation_fraction, self.config.test_fraction)
        else:
            logging.info('Loading cached primitive dataset.')
            self._dataset = dataset.CachedPrimitiveDataset(
                self.cache_data_file_path,
                self.config.num_position_bins,
                self.config.max_token_length)

        self._train_dataset, self._valid_dataset, self._test_dataset = split_dataset(
            self._dataset, self.config.validation_fraction, self.config.test_fraction)

    def _make_dataloader(self, dataset, shuffle: bool) -> torch.utils.data.DataLoader[modules.TokenInput]:
        if dataset is None:
            return None

        return torch.utils.data.DataLoader(
            dataset, self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> torch.utils.data.DataLoader[modules.TokenInput]:
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        dl1 = self._make_dataloader(self._valid_dataset, shuffle=False)

        if self._valid_dataset_no_noise is not None:
            return [self._make_dataloader(self._valid_dataset_no_noise, shuffle=False), dl1]
        else:
            return dl1

    def test_dataloader(self):
        dl1 = self._make_dataloader(self._test_dataset, shuffle=False)
        if self._test_dataset_no_noise is not None:
            return [self._make_dataloader(self._test_dataset_no_noise, shuffle=False), dl1]
        else:
            return dl1

    @property
    def cache_data_file_path(self) -> str:
        """Path to the cached token data file."""
        return _get_cache_data_path(self.config.sequence_path, self.config.num_position_bins)

    @property
    def train_dataset_size(self) -> int:
        return len(self._train_dataset)


def _list_files_in_directory(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files.sort()
    return files


class ImagePrimitiveDataModule(pytorch_lightning.LightningDataModule):
    """Datamodule encapsulating a `dataset.ImagePrimitiveDataset`.
    """
    def __init__(self, config: ImagePrimitiveDataConfig, batch_size: int=128, num_workers: int=8, seed: Optional[int]=None):
        super().__init__()

        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._dataset = None
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None
        self._valid_no_transform_dataset = None
        self._test_no_transform_dataset = None

    def prepare_data(self) -> None:
        cache_data_if_needed(self.config.sequence_path, self.config.num_position_bins)

    def setup(self):
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed)
        else:
            generator = None

        primitive_dataset = dataset.CachedPrimitiveDataset(
            self.cache_data_file_path,
            self.config.num_position_bins,
            self.config.max_token_length)

        if self.config.image_data_folder is None:
            image_data_folder = get_image_folder_from_sequence(self.config.sequence_path)
            logging.getLogger(__name__).info(
                f'Image data folder not provided. Inferred image data folder location {image_data_folder}')
            self.config.image_data_folder = image_data_folder

        image_dataset = dataset.MultiImageDataset(
            _list_files_in_directory(self.config.image_data_folder))

        if self.config.augmentation.enabled:
            import torchvision
            shift_fraction = self.config.augmentation.shift_fraction
            transform = torchvision.transforms.RandomAffine(
                self.config.augmentation.rotation,
                translate=(shift_fraction, shift_fraction),
                scale=(1 - self.config.augmentation.scale, 1 + self.config.augmentation.scale),
                shear=self.config.augmentation.shear,
                fillcolor=255)
        else:
            transform = None

        self._dataset = dataset.ImagePrimitiveDataset(
            primitive_dataset,
            image_dataset,
            image_transform=transform,
            use_noisy_images=self.config.use_noisy_images,
            generator=generator)

        self._train_dataset, self._valid_dataset, self._test_dataset = split_dataset(
            self._dataset, self.config.validation_fraction, self.config.test_fraction)

        if transform is not None:
            # When we have a transform, also create a validation dataset without transform.
            # The test dataset is always created without transform.
            _, self._valid_no_transform_dataset, self._test_no_transform_dataset = split_dataset(
                dataset.ImagePrimitiveDataset(primitive_dataset, image_dataset, use_noisy_images=self.config.use_noisy_images),
                self.config.validation_fraction,
                self.config.test_fraction)

    def _make_dataloader(self, ds, shuffle=False) -> Optional[torch.utils.data.DataLoader[dataset.ImageTokenDatum]]:
        if ds is None:
            return None

        return torch.utils.data.DataLoader(
            ds, self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False, # TODO: Error on pralexa for small sample, figure out why.
            persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> torch.utils.data.DataLoader[dataset.ImageTokenDatum]:
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader[dataset.ImageTokenDatum]]:
        dl1 = self._make_dataloader(self._valid_dataset, shuffle=False)

        if self._valid_no_transform_dataset is not None:
            return [self._make_dataloader(self._valid_no_transform_dataset, shuffle=False), dl1]
        else:
            return dl1

    def test_dataloader(self) -> Union[torch.utils.data.DataLoader, List[torch.utils.data.DataLoader]]:
        dl1 = self._make_dataloader(self._test_dataset, shuffle=False)
        if self._test_no_transform_dataset is not None:
            return [self._make_dataloader(self._test_no_transform_dataset, shuffle=False), dl1]
        else:
            return dl1

    @property
    def train_dataset_size(self) -> int:
        return len(self._train_dataset)

    @property
    def cache_data_file_path(self) -> str:
        """Path to the cached token data file."""
        return _get_cache_data_path(self.config.sequence_path, self.config.num_position_bins)
