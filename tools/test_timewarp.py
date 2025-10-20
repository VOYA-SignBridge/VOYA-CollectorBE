import sys, os
sys.path.insert(0, os.path.abspath('.'))
from backend.app.processing.augmenter import time_warp
import numpy as np

for f in [1.2, 0.8, 1.0]:
    x = np.random.rand(60,226).astype(np.float32)
    y = time_warp(x, factor=f)
    print('factor', f, '->', y.shape)
