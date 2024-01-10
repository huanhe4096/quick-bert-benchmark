#%% load libs
import time
import numpy as np
from openTSNE import TSNE
print('* loaded libs')

import tracemalloc

tracemalloc.start()
# Your memory-intensive code here

#%% create sample
dim = 384
n_samples = int(1 * 1000 * 1000)
X = np.random.rand(n_samples, dim)
print('* created sample', X.shape)


#%% create tsne
tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)
print('* created tsne')

#%% run tsne
start_time = time.time()
embds = tsne.fit(X)
total_time = time.time() - start_time
print(f'* Time(seconds): {total_time:.4f}')
print(f'* ({total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.2f} seconds)')


#%% check memory
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

for stat in top_stats[:10]:  # Print the top 10 memory-consuming lines
    print(stat)


#%% visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(embds[:, 0], embds[:, 1], s=1, alpha=0.5, c='b')
ax.set_title("Random data embedded into two dimensions by t-SNE", fontsize=18)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)

fig.show()