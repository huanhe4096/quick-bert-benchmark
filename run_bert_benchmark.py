#%% load libs
import time
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
print('* loaded libs')

# create a decorator to measure time
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        total_time = time.time() - start_time
        # display the time in x hour, x minutes, x seconds format
        print(f'* [{func.__name__}] {len(args[0])} records took {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.2f} seconds')
        # display the time in seconds
        print(f'* Time(seconds): {total_time:.4f}')
        # display the time per 1000 samples
        print(f'* Time per 1k(seconds): {1000 * total_time / len(args[0]):.4f}')
        # display the number of samples per second
        print(f'* N per second: {len(args[0]) / total_time:.4f}')
        return result
    return wrapper


# create a function to benchmark the model on given texts
@timeit
def benchmark(texts, model):
    # encode the texts
    encoded_data = model.encode(
        texts,
        show_progress_bar=True,
    )
    return encoded_data

#%% load model
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
print('* loaded model')

#%% prepare data
dataset = load_dataset("OxAISH-AL-LLM/pubmed_20k_rct")

#%% run benchmark
benchmark(dataset['train']['text'], model)