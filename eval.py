import os
import sys
import torch
import logging
import sklearn
from os.path import join
from datetime import datetime
import test
import util
import commons
import datasets_ws
import network
import parser
import warnings
from multiprocessing import freeze_support

warnings.filterwarnings('ignore')


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")

    model = network.HamVPRNet()
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    state_dict = torch.load('./weights/HAM-VPR.pth')
    model.load_state_dict(state_dict)

    if args.pca_dim == None:
        pca = None
    else:
        full_features_dim = args.features_dim
        args.features_dim = args.pca_dim
        pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    logging.info(f"Test set: {test_ds}")

    recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")


if __name__ == '__main__':
    freeze_support()
    main()
