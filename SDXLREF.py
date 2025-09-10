import os
import pandas as pd
import multiprocessing
from multiprocessing import Process
from tqdm import tqdm
import time
import torch

# ======= 固定参数配置 =======
BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
CACHE_DIR = "/projects/0/prjs1364/xwang/cache"
SAVE_DIR = "/projects/0/prjs1364/xwang/code/Mediaeval2025/diffusionxl_refiner_gen_images"
CSV_PATH = "/projects/0/prjs1364/xwang/code/Mediaeval2025/newsimages_25_v1.1/newsarticles.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

# ======= 每个 GPU 的子进程 =======
def run_worker(gpu_id, rows):
    import os
    import torch
    from diffusers import DiffusionPipeline
    import gc

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.set_device(gpu_id)

    print(f"[GPU {gpu_id}] Starting with {len(rows)} prompts...")
    print(f"[GPU {gpu_id}] Device: {torch.cuda.get_device_name(gpu_id)}")

    start_time = time.time()

    # ====== 加载 base 和 refiner pipeline 到指定 GPU ======
    base = DiffusionPipeline.from_pretrained(
        BASE_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(f"cuda:{gpu_id}")

    # ✅ 对 base unet 编译加速
    # base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

    refiner = DiffusionPipeline.from_pretrained(
        REFINER_ID,
        cache_dir=CACHE_DIR,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(f"cuda:{gpu_id}")

    base.enable_attention_slicing()
    refiner.enable_attention_slicing()

    n_steps = 40
    high_noise_frac = 0.8  # 80% denoising in base, 20% in refiner

    for image_id, title in tqdm(rows, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            img_start = time.time()

            # base 生成 latent
            latent = base(
                prompt=title,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type="latent",
            ).images

            # refiner 完成图像生成
            image = refiner(
                prompt=title,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=latent,
            ).images[0]

            image.save(os.path.join(SAVE_DIR, f"{image_id}.jpg"))

            print(f"[GPU {gpu_id}] Finished {image_id} in {time.time() - img_start:.2f}s")

            # 显存释放
            del image, latent
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"[GPU {gpu_id}] Error on {image_id}: {e}")

    print(f"[GPU {gpu_id}] Completed {len(rows)} in {(time.time() - start_time)/60:.2f} minutes")

# ======= 按 GPU 数量均匀分配数据 =======
def split_chunks(data, num_chunks):
    k, m = divmod(len(data), num_chunks)
    return [data[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_chunks)]

# ======= 启动主程序 =======
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

    print("Finished", SAVE_DIR)