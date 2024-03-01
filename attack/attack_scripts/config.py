"""Config file for automatic code running
Assign some hyper-parameters, e.g. batch size for attack
"""
BEST_WEIGHTS = {
    # trained on standard mn40 dataset
    'mn40': {
        1024: {
            'dgcnn':'pretrain/mn40/dgcnn.pth',
            'pointnet':'pretrain/mn40/pointnet.pth',
            'pct':'pretrain/mn40/pct.pth',
            'gdanet':'pretrain/mn40/gdanet.pth'
        },
    },
    'sp': {
        1024: {
            'dgcnn':'pretrain/sp/dgcnn.pth',
            'pointnet':'pretrain/sp/pointnet.pth',
            'pct':'pretrain/sp/pct.pth',
            'gdanet':'pretrain/sp/gdanet.pth',
        },
    },
}

# PU-Net trained on Visionair with 1024 input point number, up rate 4
PU_NET_WEIGHT = 'defense/DUP_Net/pu-in_1024-up_4.pth'

# Note: the following batch sizes are tested on a RTX 2080 Ti GPU
# you may need to slightly adjust them to fit in your device

# max batch size used in testing model accuracy
MAX_TEST_BATCH = {
    1024: {
        'pointnet': 512,
        'dgcnn': 96,
        'pct': 512,
        'gdanet':96,
    },
}

# max batch size used in testing model accuracy with DUP-Net defense
# since there will be 4x points in DUP-Net defense results
MAX_DUP_TEST_BATCH = {
    1024: {
        'pointnet': 160,
        'dgcnn': 26,
        'pct': 512,
        'gdanet':96,
    },
}

# max batch size used in Add attack
MAX_ADD_BATCH = {
    1024: {
        'pointnet': 256,
        'dgcnn': 32,
        'pct':32,
        'gdanet':32,
    },
}

# max batch size used in Add Cluster attack
MAX_ADD_CLUSTER_BATCH = {
    1024: {
        'pointnet': 128,
        'dgcnn': 32,
        'pct':32,
        'gdanet':32,
    },
}

# max batch size used in Add Object attack
MAX_ADD_OBJECT_BATCH = {
    1024: {
        'pointnet': 128,
        'dgcnn': 42,
        'pct':42,
        'gdanet':42,
    },
}

# max batch size used in Drop attack
MAX_DROP_BATCH = {
    1024: {
        'pointnet': 256,
        'dgcnn': 32,
        'pct':32,
        'gdanet': 64,
    },
}

MAX_AdvPC_BATCH = {
    1024: {
        'pointnet': 256,
        'dgcnn': 32,
        'pct':32,
        'gdanet': 64,
    },
}

MAX_PERTURB_BATCH = {
    1024: {
        'pointnet': 248,
        'dgcnn': 52,
    },
}

# max batch size used in kNN attack
MAX_KNN_BATCH = {
    1024: {
        'pointnet': 248,
        'dgcnn': 42,
    },
}