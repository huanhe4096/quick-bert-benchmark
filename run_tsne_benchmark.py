#%% load libs
import argparse
import time
import numpy as np
from openTSNE import TSNE
print('* loaded libs')

import tracemalloc


#%% create sample
def create_sample(n_samples=10000, dim=384):
    X = np.random.rand(n_samples, dim)
    print('* created sample', X.shape)
    return X


#%% create tsne
def run_tsne(X, n_jobs=4):
    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=n_jobs,
        random_state=42,
        verbose=True,
    )
    print('* created tsne')

    start_time = time.time()
    embds = tsne.fit(X)
    total_time = time.time() - start_time
    print(f'* Time(seconds): {total_time:.4f}')
    print(f'* ({total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.2f} seconds)')


def main(n_samples=10000, dim=384, n_jobs=4):
    tracemalloc.start()
    # Your memory-intensive code here
    X = create_sample(n_samples, dim)
    run_tsne(X, n_jobs)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    for stat in top_stats[:10]:  # Print the top 10 memory-consuming lines
        print(stat)


def vis(embds):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(embds[:, 0], embds[:, 1], s=1, alpha=0.5, c='b')
    ax.set_title("Random data embedded into two dimensions by t-SNE", fontsize=18)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)

    fig.show()

#%% main
if __name__ == '__main__':
    # add argument parser for n_samples and dim
    parser = argparse.ArgumentParser(description='run tsne benchmark')
    parser.add_argument('--n_samples', type=int, default=10000, help='number of samples')
    parser.add_argument('--dim', type=int, default=384, help='dimension of samples')
    parser.add_argument('--n_jobs', type=int, default=4, help='number of jobs')

    args = parser.parse_args()
    print('* args:', args)

    main(
        args.n_samples,
        args.dim,
        args.n_jobs,
    )