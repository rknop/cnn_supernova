## Code taken from Steve Farrel, Nov 1, 2018
"""
Hardware/device configuration
"""
# System
import os
# Externals
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
# Default settings are for Haswell
def configure_session(intra_threads=32, inter_threads=2,
                      blocktime=1, affinity='granularity=fine,compact,1,0'):
    """Sets the thread knobs in the TF backend"""
    os.environ['KMP_BLOCKTIME'] = str(blocktime)
    os.environ['KMP_AFFINITY'] = affinity
    os.environ['OMP_NUM_THREADS'] = str(intra_threads)
    
    config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=inter_threads,
        intra_op_parallelism_threads=intra_threads
    )
    keras.backend.set_session(tf.compat.v1.Session(config=config))
