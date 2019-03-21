#copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
Vehicle = {
    'SIZE': [320,320],
    'REFINE': True,
    'CONV_BODY': 'RefineResnet18',
    'NUM_CLASSES': 4,
    'ARM_CHANNELS': [128, 256, 512, 256],
    'ODM_CHANNELS': [256, 256, 256, 256],
    'FEATURE_MAPS': [[40, 40], [20, 20], [10, 10], [5, 5]],
    'STEPS': [[8., 8.], [16., 16.], [32., 32.], [64., 64.]],
    'STAGE_STEPS': [8., 16., 32., 64.],
    'MIN_SIZES': [30., 64., 128., 256.],
    'USE_MAX_SIZE': False,
    'MAX_SIZES': [64., 128., 256., 315.],
    'ASPECT_RATIOS' : [[2, 0.5], [2, 0.5], [2, 0.5], [2, 0.5]],
    'NUM_ANCHORS': [3, 3, 3, 3],
    'VARIANCE' : [0.1, 0.2],
    'CLIP': True,
    'FLIP': False,
    'OFFSET': 0.5,
    'OVERLAP_THRESH': 0.5,
    'BG_LABEL': 0,
    'NEG_OVERLAP': 0.5,
    'TOP_K': 200,
    'CONF_THRESH': 0.45,
    'NMS_THRESH' : 0.3
}

