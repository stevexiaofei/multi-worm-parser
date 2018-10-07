import argparse
import os
import scipy.misc
import numpy as np

from model import Singleout_net
from dataprovider import data_provider
import tensorflow as tf
from utils import process_config
cfg= process_config('exp1//config.cfg')
gene = data_provider(cfg)
def main(_):
    if not os.path.exists(os.path.join(cfg['exp_name'],'checkpoint')):
        os.makedirs(os.path.join(cfg['exp_name'],'checkpoint'))
    if not os.path.exists(os.path.join(cfg['exp_name'],'sample')):
        os.makedirs(os.path.join(cfg['exp_name'],'sample'))
    if not os.path.exists(os.path.join(cfg['exp_name'],'test')):
        os.makedirs(os.path.join(cfg['exp_name'],'test'))

    with tf.Session() as sess:
        model = Singleout_net(sess,cfg,gene,image_size=cfg['fine_size'], batch_size=cfg['batch_size'],
                   output_size=cfg['fine_size'], dataset_name=cfg['dataset_name'],
                        checkpoint_dir=cfg['checkpoint_dir'], sample_dir=cfg['sample_dir'])

        if cfg['phase'] == 'train':
            model.train(cfg)
        else:
            model.test(cfg)

if __name__ == '__main__':
    tf.app.run()
