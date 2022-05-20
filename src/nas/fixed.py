import json
from loguru import logger
from pathlib import Path
from typing import Union, Dict, Any

from utils.utils import ContextStack
from supernets.search_space_config import expert_archs

def fixed_arch(fixed_arch: Union[str, Path, Dict[str, Any]]):
    """
    Load architecture from ``fixed_arch`` and apply to model. This should be used as a context manager. For example,

    .. code-block:: python

        with fixed_arch('/path/to/export.json'):
            model = Model(3, 224, 224)

    Parameters
    ----------
    fixed_arc : str, Path or dict
        Path to the JSON that stores the architecture, or dict that stores the exported architecture.
    verbose : bool
        Print log messages if set to True

    Returns
    -------
    ContextStack
        Context manager that provides a fixed architecture when creates the model.
    """

    if isinstance(fixed_arch, (str, Path)):
        with open(fixed_arch) as f:
            fixed_arch = json.load(f)

    logger.info(f'Fixed architecture: {fixed_arch}')

    return ContextStack('fixed', fixed_arch)


def merged_fix_arch(searched_arch):
    """Load and merge architectures from ``searched_arch`` and apply to model.
    This should be used as a context manager. For example,

    .. code-block:: python

        with merged_fix_arch('/path/to/xxx.stats'):
            model = Model(3, 224, 224)
            
    This method used to merge experts searched by MOO-NAS and constructs a MMoE structure.

    Parameters
    ----------
    searched_arch : str, Path or dict
        Path to the JSON that stores the architecture, or dict that stores the exported architecture.
    verbose : bool
        Print log messages if set to True

    Returns
    -------
    ContextStack
        Context manager that provides a fixed architecture when creates the model.
    """
    if isinstance(searched_arch, (str, Path)):
        with open(searched_arch) as f:
            searched_arch = json.load(f)

    experts = searched_arch['candidates']
    fixed_arch = {}
    for i, expert in enumerate(experts):
        expert = expert[0]
        for layer, block in expert.items():
            tokens = layer.split('_')
            tokens[1] = str(i)
            layer = '_'.join(tokens)
            block = expert_archs['ops'][block]
            fixed_arch[layer] = block

    logger.info(f'Fixed architecture: {fixed_arch}')
    
    return ContextStack('fixed', fixed_arch)