import os

import pickle

from loguru import logger
from torch.utils.data import DataLoader

from datasets.dataset import Dataset
from utils.utils import PROJECT_PATH

from prefetch_generator import BackgroundGenerator
from tqdm.auto import tqdm


class DataProvider:
    """circulating generate batch of data from dataloader"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(dataloader)
        
    def next(self):
        try:
            _, data = next(self.iterator)
        except:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data


def create_dataset(config):
    """Create dataset.

    Args:
        config (Config): configurations.
    Returns:
        Dataset: Consturcted dataset.
    """
    default_file = os.path.join(PROJECT_PATH, config.checkpoint_dir, 'datasets', f'{config.dataset}-splited-datasets.pth')
    splited_datasets_file = config.splited_datasets_file or default_file
    if os.path.exists(splited_datasets_file):
        with open(splited_datasets_file, 'rb') as f:
            datasets = pickle.load(f)
        
        logger.info(f'Dataset Name: {datasets[0].dataset_name}, '
                    f'Train data number: {len(datasets[0])}, '
                    f'Validation data number: {len(datasets[1])}, '
                    f'Test data number: {len(datasets[2])}, '
                    f'Token features dim: {datasets[0].field_token_dims}, '
                    f'Token sequence features dim: {datasets[0].field_token_seq_dims}, '
                    f'Float feature field num: {len(datasets[0].float_fields)}')    
        
        return datasets
    
    default_file = os.path.join(PROJECT_PATH, config.checkpoint_dir, 'datasets', f'{config.dataset}-dataset.pth')
    dataset_file = config.dataset_file or default_file
    if os.path.exists(dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
        datasets = dataset.build()
        
        logger.info(f'Dataset Name: {dataset.dataset_name}, '
                f'Train data number: {len(datasets[0])}, '
                f'Validation data number: {len(datasets[1])}, '
                f'Test data number: {len(datasets[2])}, '
                f'Token features dim: {dataset.field_token_dims}, '
                f'Token sequence features dim: {dataset.field_token_seq_dims}, '
                f'Float feature field num: {len(dataset.float_fields)}')
        
        return datasets
    
    dataset = Dataset(config)
    datasets = dataset.build()
    if config.save_dataset:
        dataset.save()
        with open(splited_datasets_file, 'wb') as f:
            pickle.dump(datasets, f)
    # print(dataset.field_token_nums, dataset.field_token_seq_nums)
    
    logger.info(f'Dataset Name: {dataset.dataset_name}, '
                f'Data number: {len(dataset)}, '
                f'Train data number: {len(datasets[0])}, '
                f'Validation data number: {len(datasets[1])}, '
                f'Test data number: {len(datasets[2])}, '
                f'Token features dim: {dataset.field_token_dims}, '
                f'Token sequence features dim: {dataset.field_token_seq_dims}, '
                f'Float feature field num: {len(dataset.float_fields)}')
    
    return datasets
    

def get_dataloader(datasets, config, num_workers=1):
    """Get Dataloaders

    Args:
        config (Config): configurations.
    Returns:
        DataLoader (train, validate, test): Consturcted DataLoaders.
    """
    train_set, validate_set, test_set = datasets

    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=config.eval_batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.eval_batch_size, num_workers=num_workers, pin_memory=True)
    
    return train_loader, validate_loader, test_loader

if __name__ == '__main__':
    from config.config import config
    train_loader, validate_loader, test_loader = get_dataloader(config)
    train_queue = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
    for i, data in train_queue:
        print(data)