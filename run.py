import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import client
import config
import logging
import server

import tensorflow as tf
print(tf.test.is_gpu_available())
'''sess_config = tf.compat.v1.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.70
sess = tf.compat.v1.Session(config=sess_config)'''

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./dqn_noniid.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()
case_name = args.config.split('/')[-1].split('.')[0]
print("case_name:", case_name)

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config)

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config, case_name),
        "accavg": server.AccAvgServer(fl_config, case_name),
        "directed": server.DirectedServer(fl_config, case_name),
        "kcenter": server.KCenterServer(fl_config, case_name),
        "kmeans": server.KMeansServer(fl_config, case_name),
        "magavg": server.MagAvgServer(fl_config, case_name),
        "dqn": server.DQNServer(fl_config, case_name), # DQN inference server 
        "dqntrain": server.DQNTrainServer(fl_config, case_name), # DQN train server
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Delete global model
    #os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    main()
