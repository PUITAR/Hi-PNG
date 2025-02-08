def create_dataset(dataset):
  from os import system
  
  system(f"""
  cd .. && \
  python create_dataset.py --dataset {dataset} \
    --train_distr uniform --train_left 0 --train_right 1000 \
    --test_distr uniform --test_left 0 --test_right 1000 \
    --k 100 --num_threads 48
  """)

  # system(f"""
  # cd .. && \
  # python create_dataset.py --dataset {dataset} \
  #   --train_distr normal --train_mean 0 --train_std 1 \
  #   --test_distr normal --test_mean 0 --test_std 1 \
  #   --k 100 --num_threads 48
  # """)

  # system(f"""
  # cd .. && \
  # python create_dataset.py --dataset {dataset} \
  #   --train_distr poisson --train_lam 100 \
  #   --test_distr poisson --test_lam 100 \
  #   --k 100 --num_threads 48
  # """)
  

def pareto_frontier(perf, lowest_recall):
  change = True
  while change:
    change = False
    for i in range(len(perf)):
      bk = False
      for j in range(len(perf)):
        if i != j and perf[i]['recall'] >= perf[j]['recall'] and perf[i]['qps'] >= perf[j]['qps']:
          bk = change = True
          break
      if bk:
        perf.remove(perf[j])
        break
  perf = sorted(perf, key=lambda x: x['recall'])
  perf = [item for item in perf if item['recall'] >= lowest_recall]
  return perf



def plot_qps_recall_curve(perf_pf_path, perf_ct_path, fname, underlying_graph, lowest_recall = 0):
  import json
  with open(perf_pf_path, "r") as f:
    perf_pf = pareto_frontier(json.load(f), lowest_recall)
  with open(perf_ct_path, "r") as f:
    perf_ct = pareto_frontier(json.load(f), lowest_recall)
  recall_pf = [item["recall"] for item in perf_pf]
  qps_pf = [item["qps"] for item in perf_pf]
  recall_ct = [item["recall"] for item in perf_ct]
  qps_ct = [item["qps"] for item in perf_ct]
  import matplotlib.pyplot as plt
  plt.figure(figsize=(4, 3))
  plt.plot(recall_pf, qps_pf, marker='s', color='tomato', linewidth=1.2, linestyle='dashed', markersize=3)
  plt.plot(recall_ct, qps_ct, marker='o', color='deepskyblue', linewidth=1.2, linestyle='dashed', markersize=3)
  plt.xlabel('Recall')
  plt.ylabel('QPS') 
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  plt.grid(True)
  gn = {"hnsw": "HNSW", "nsg": "NSG", "hcnng": "HCNNG", "vamana": "Vamana"}[underlying_graph]
  plt.legend([f'{gn}', f'IPNG-{gn}'], loc='center', bbox_to_anchor=(0.51, 1.1), ncol = 2, frameon=False)
  if fname is not None:
    plt.savefig(fname, bbox_inches='tight')
  # plt.show()