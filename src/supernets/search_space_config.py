bottom_archs = {
    'layer_num': 1,
    'ops': [
        'skipconnect',

        'InnerProduct', 'OuterProduct',
        'Crossnet-1', 'Crossnet-2', 'Crossnet-3', 'Crossnet-4',

        "SelfAttention-1", "SelfAttention-2", "SelfAttention-4", "SelfAttention-8"
    ],
    'block_in_dim': 64,
    'block_out_dim': 64,
    'embedding_dim': None,  # need to set
    'num_fields': None,  # need to set
}

expert_archs = {
    'layer_num': 3,
    'ops': [
        'skipconnect',
        'GatingLayer',
        "MLP-16", "MLP-32", "MLP-64", "MLP-128", "MLP-256", "MLP-512", "MLP-1024",
        'Conv1D-3', 'Conv1D-5', 'Conv1D-7',
    ],
    'block_in_dim': 64,
    'block_out_dim': 64,
}