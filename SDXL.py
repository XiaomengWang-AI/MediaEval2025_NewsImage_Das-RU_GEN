import os
import pandas as pd
import multiprocessing
from multiprocessing import Process
from tqdm import tqdm
import time
import torch
from PIL import Image

# 参数配置
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CACHE_DIR = "/projects/0/prjs1364/xwang/cache"
SAVE_DIR = "/projects/0/prjs1364/xwang/code/Mediaeval2025/diffusionxl_gen_images_seed_png"
CSV_PATH = "/projects/0/prjs1364/xwang/code/Mediaeval2025/newsimages_25_v1.1/newsarticles.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

generator = torch.manual_seed(12345)
def run_worker(gpu_id, rows):
    import os
    import torch
    from diffusers import AutoPipelineForText2Image
    import gc
    import sys

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.set_device(gpu_id)

    print(f"[GPU {gpu_id}] Starting with {len(rows)} prompts...")
    print(f"[GPU {gpu_id}] Device: {torch.cuda.get_device_name(gpu_id)}")

    start_time = time.time()

    pipeline = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(f"cuda:{gpu_id}")

    for image_id, title in tqdm(rows, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            img_start = time.time()
            image = pipeline(prompt=title, height=1024, width=1024,num_inference_steps=40, guidance_scale=7.5).images[0]
            image.save(os.path.join(SAVE_DIR, f"{image_id}.jpg"))
            print(f"[GPU {gpu_id}] Finished {image_id} in {time.time() - img_start:.2f}s")

            # 显存清理
            del image
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"[GPU {gpu_id}] Error on {image_id}: {e}")

    print(f"[GPU {gpu_id}] Completed {len(rows)} in {(time.time() - start_time)/60:.2f} minutes")

def split_chunks(data, num_chunks):
    k, m = divmod(len(data), num_chunks)
    return [data[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_chunks)]

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    NUM_GPUS = torch.cuda.device_count()
    print(f"Detected {NUM_GPUS} GPUs.")

    df = pd.read_csv(CSV_PATH)
    items = list(zip(df['image_id'], df['article_title']))
    chunks = split_chunks(items, NUM_GPUS)

    processes = []
    for gpu_id in range(NUM_GPUS):
        p = Process(target=run_worker, args=(gpu_id, chunks[gpu_id]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All images have been generated and saved.")