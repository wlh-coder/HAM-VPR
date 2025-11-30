import argparse
import os

import torch


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="HAM-VPR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size in triplets (query+pos+neg). Recommended: 4-16, must be divisible by (1+negs_num_per_query)")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Inference batch size (for caching/testing). Suggested: 2-4x train_batch_size")
    parser.add_argument("--criterion", type=str, default='triplet',
                        help="Loss function: triplet (standard) | sare_joint (joint similarity) | sare_ind (independent similarity)")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Margin for triplet loss. Range: 0.05-0.3 (higher values enforce stricter separation)")
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="Maximum training epochs (may stop earlier via early stopping)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement before stopping)")
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="Initial learning rate. Suggested: 1e-5 to 1e-4 (smaller for larger models)")
    parser.add_argument("--optim", type=str, default="adamW",
                        help="Optimizer: adam (adaptive momentum) | sgd (with momentum) | adamW (corrected weight decay)")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="Feature cache refresh frequency (in queries). Lower=frequent updates but slower")
    parser.add_argument("--negs_num_per_query", type=int, default=2,
                        help="Negative samples per query. Higher values increase difficulty but may improve discrimination")
    parser.add_argument("--mining", type=str, default="partial",
                        help="Negative mining strategy: partial (semi-hard) | full (hardest) | random | msls_weighted (dataset-aware)")
    parser.add_argument("--l2", type=str, default="before_pool",
                        help="L2 normalization timing: before_pool | after_pool | none")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="PCA reduction dimension (None to disable).")
    parser.add_argument("--datasets_folder", type=str, default='./Data/4_Train_data',
                        help="Root directory for datasets (checks DATASETS_FOLDER env if not specified)")
    parser.add_argument("--save_dir", type=str, default="./logs/",
                        help="Experiment output directory (auto-creates timestamped subfolder)")
    args = parser.parse_args()

    if args.datasets_folder == None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")


    return args
