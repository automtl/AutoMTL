from enum import Enum

class FeatureType(Enum):
    """Type of features.
    
    - ``TOKEN``: Token features like user_id and item_id
    - ``FLOAT``: Float features like rating and timestamp.
    - ``TOKEN_SEQ``: Token sequence features like review.
    - ``FLOAT_SEQ``: Float sequence features like pretrained vector.
    """
    
    LABEL = 'label'
    TOKEN = 'token'
    FLOAT = 'float'
    TOKEN_SEQ = 'token_seq'
    FLOAT_SEQ = 'float_seq'
    
class FeatureSource(Enum):
    """Source of features

    - ``INTERACTION``: Interaction features
    - ``USER``: User features
    - ``ITEM``: Item features
    - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``
    - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``
    """
    
    INTERACTION = 'inter'
    USER = 'user'
    ITEM = 'item'
    USER_ID = 'user_id'
    ITEM_ID = 'item_id'