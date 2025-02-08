import argparse

from data_collect.rawdata import DATASETS, get_dataset_fn
from data_collect.binary import fvecs_read, fvecs_save
from data_collect.interval import DISTRIBUTION, create_interval, distr_name
import os, glob

# You must use in the root of this project
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
  # parser.add_argument("--output", required=True)

  parser.add_argument("--train_distr", choices=DISTRIBUTION, required=False, default="uniform", 
                      help="""The distribution of the interval. 
                              Default is uniform.""")
  parser.add_argument("--train_left", required=False, type=float, default=0, 
                      help="""The left bound of the uniform distribution's left bound (inclusive).""")
  parser.add_argument("--train_right", required=False, type=float, default=1,
                      help="""The right bound of the uniform distribution's right bound (exclusive).""")
  parser.add_argument("--train_mean", required=False, type=float, default=0,
                      help="""The mean of the normal distribution.""")
  parser.add_argument("--train_std", required=False, type=float, default=1,
                      help="""The standard deviation of the normal distribution.""")
  parser.add_argument("--train_lam", required=False, type=float, default=1,
                      help="""The lambda of the possion distribution.""")
  # parser.add_argument("--train_length", required=False, type=float, default=-1, 
  #                     help="""max length of the train interval""")
  
  parser.add_argument("--test_distr", choices=DISTRIBUTION, required=False, default="uniform", 
                      help="""The distribution of the interval. 
                              Default is uniform.""")
  parser.add_argument("--test_left", required=False, type=float, default=0, 
                      help="""The left bound of the uniform distribution's left bound (inclusive).""")
  parser.add_argument("--test_right", required=False, type=float, default=1,
                      help="""The right bound of the uniform distribution's right bound (exclusive).""")
  parser.add_argument("--test_mean", required=False, type=float, default=0,
                      help="""The mean of the normal distribution.""")
  parser.add_argument("--test_std", required=False, type=float, default=1,
                      help="""The standard deviation of the normal distribution.""")
  parser.add_argument("--test_lam", required=False, type=float, default=1,
                      help="""The lambda of the possion distribution.""")

  parser.add_argument("--k", required=False, type=int, default=10, 
                      help="""The number of top k.
                              Default is 10.""")
  parser.add_argument("--num_threads", required=False, type=int, default=1,
                      help="""The number of process.""")
  # parser.add_argument("--test_length", required=False, type=float, default=-1, 
  #                     help="""max length of the test interval""")

  args = parser.parse_args()

  # Download raw data if not exist
  if glob.glob(f"data/{args.dataset}*"):
    print("Raw data already exist.")
  else:
    print("Downloading raw data...")
    fn = get_dataset_fn(args.dataset)
    DATASETS[args.dataset](fn)

  # print("Downloading raw data...")
  # fn = get_dataset_fn(args.dataset)
  # DATASETS[args.dataset](fn)

  if args.dataset == 'us-stock-384-euclidean':
    print("real world dataset")
    train_dn = test_dn = ""
  else:
    train_dn = distr_name(args.train_distr, left=args.train_left, right=args.train_right, mean=args.train_mean, std=args.train_std, lam=args.train_lam)
    test_dn = distr_name(args.test_distr, left=args.test_left, right=args.test_right, mean=args.test_mean, std=args.test_std, lam=args.test_lam)
  
  # Calculate Groundtruth
  base_file = os.path.join("data", ".".join([args.dataset, "train", "fvecs"]))
  test_file = os.path.join("data", ".".join([args.dataset, "test", "fvecs"]))
  base_itv_file = os.path.join("data", ".".join([args.dataset, train_dn, "train", "itv"]))
  test_itv_file = os.path.join("data", ".".join([args.dataset, test_dn, "test", "itv"]))
  gt_file = os.path.join("data", ".".join([args.dataset, train_dn, test_dn, "gt"]))

  if not os.path.exists(base_itv_file):
    print("Generating Train Interval...")
    fvecs_save(base_itv_file, create_interval(fvecs_read(base_file).shape[0], args.train_distr, left=args.train_left, right=args.train_right, mean=args.train_mean, std=args.train_std, lam=args.train_lam))
  else:
    print("Train interval already exist.")

  if not os.path.exists(test_itv_file):
    print("Generating Test Interval...")
    fvecs_save(test_itv_file, create_interval(fvecs_read(test_file).shape[0], args.test_distr, left=args.test_left, right=args.test_right, mean=args.test_mean, std=args.test_std, lam=args.test_lam))
  else:
    print("Test interval already exist.")

  if not os.path.exists(gt_file):
    print(f"Generating Groundtruth {gt_file}...")
    os.system(f"./bin/groundtruth \
              {base_file} \
              {test_file} \
              {base_itv_file} \
              {test_itv_file} \
              {gt_file} \
              {args.k} \
              {args.num_threads}")
  else:
    print(f"Groundtruth {gt_file} already exist.")