import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))

import argparse
import ast
from loguru import logger

from supernets.supernet import Supernet

from config.config import get_config, parse_cli_to_yaml, parse_yaml, merge, Config
from datasets.dataset_utils import create_dataset, get_dataloader
from trainer.trainer import Trainer
from utils.utils import PROJECT_PATH, dict2str, set_logger, ensure_dir, set_seed
from nas.fixed import fixed_arch

def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config_path", type=str, default=os.path.join(PROJECT_PATH, 'config', "config.yaml"),
                        help="Config file path")
    return parser

def get_model(dataset, config):
    """Get the evaluate model.

    Args:
        name (str): model name
        dataset (Dataset): dataset object
        config (Config): global configuration
    """
    logger.info(f'Get Supernet for {config.model}.')
    return Supernet(dataset, config)

@logger.catch
def main():
    parser = get_args()
    config = get_config(parser)
    set_seed()
    
    log_root_path = os.path.join(PROJECT_PATH, config.log_dir, 'automtl', config.dataset)
    ensure_dir(log_root_path)
    
    final_architecture_path = os.path.join(PROJECT_PATH, config.checkpoint_dir, 'automtl', config.dataset,
                                           f'{config.expert_num}_{config.expert_layer_num}_'
                                           f'{config.auxiliary_loss_weight}_'
                                           f'{config.dropout}_{config.drop_path}'
                                           )
    ensure_dir(final_architecture_path)
    final_architecture_path = os.path.join(final_architecture_path, 'final_architecture.json')
    
    log_file = os.path.join(
        log_root_path,
        f'log_{config.dataset}_automtl_softmax_{config.expert_num}_{config.expert_layer_num}_{config.chosen_experts}_'
        f'{config.auxiliary_loss_weight}_'
        f'{config.dropout}_{config.drop_path}'
        f'_retrain.log'
    )
    set_logger(log_file)
    
    logger.info(config)
    
    datasets = create_dataset(config)
    
    with fixed_arch(final_architecture_path):
        model = get_model(datasets[0], config)
        
    train_loader, val_loader, test_loader = get_dataloader(datasets, config)
    
    trainer = Trainer(config, model)
    trainer.fit(train_loader, val_loader)
    result = trainer.evaluate(test_loader, is_test=True)
    logger.info(f'Model={config.model}, test result: {dict2str(result)}')

if __name__ == '__main__':
    main()
