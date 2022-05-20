import numpy as np


from nas.choicer import LayerChoice, ExpertChoice

def replace_layer_choice(root_module, init_fn, modules=None, type_name=LayerChoice):
    """Replace layer choice modules with modules that are initiated with init_fn
    
    Args:
        root_module (nn.Module): Root module to traverse.
        init_fn (Callable): Initialzing function
        modules (dict, optional): update the replaced modules into the dict and check duplicate if provided.
            Defaults to None.
    Returns:
        List[Tuple[str, nn.Module]]
            A list from layer choice keys (names) and replaced modules.
    """
    if modules is None:
        modules = []
    def apply(m):
        for name, child in m.named_children():
            if isinstance(child, type_name):
                setattr(m, name, init_fn(child))
                modules.append((child.label, getattr(m, name)))
            else:
                apply(child)
    apply(root_module)
    return modules

def replace_expert_choice(root_module, init_fn, modules=None, type_name=ExpertChoice):
    """Replace expert choice modules with modules that are initiated with init_fn
    
    Args:
        root_module (nn.Module): Root module to traverse.
        init_fn (Callable): Initialzing function
        modules (dict, optional): update the replaced modules into the dict and check duplicate if provided.
            Defaults to None.
    Returns:
        List[Tuple[str, nn.Module]]
            A list from layer choice keys (names) and replaced modules.
    """
    if modules is None:
        modules = []
    def apply(m):
        for name, child in m.named_children():
            if isinstance(child, type_name):
                setattr(m, name, init_fn(child))
                modules.append((child.label, getattr(m, name)))
            else:
                apply(child)
    apply(root_module)
    return modules
