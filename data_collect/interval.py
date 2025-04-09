import numpy as np

DISTRIBUTION: list[str] = ["uniform", "normal","poisson"]

def create_interval(n: int, distribution: str, left=None, right=None, mean=None, std=None, lam=None, rf: bool = False) -> np.ndarray:
  if distribution == "uniform":
    if left >= right:
      raise ValueError("The left bound must be less than the right bound.")
    itv = np.random.uniform(low=left, high=right, size=(n,2))
  elif distribution == "normal":
    itv = np.random.normal(loc=mean, scale=std, size=(n,2))
  elif distribution == "poisson":
    itv = np.random.poisson(lam=lam, size=(n,2))
  else:
    raise ValueError("Invalid distribution.")
  if rf == True:
    itv[:,1] = itv[:,0]
  else:
    itv = np.sort(itv, axis=1)
  return itv

def distr_name(distribution, left, right, mean, std, lam) -> str:
  if distribution == "uniform":
    name = f"uniform-{left}-{right}"
  elif distribution == "normal":
    name = f"normal-{mean}-{std}"
  elif distribution == "poisson":
    name = f"poisson-{lam}"
  else:
    raise ValueError("Invalid distribution.")
  return name