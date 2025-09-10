from PIL import Image
import os


# ===== 配置路径 =====
input_folder = "/projects/0/prjs1364/xwang/code/Mediaeval2025/diffusionxl_gen_images"
output_folder = "/projects/0/prjs1364/xwang/code/Mediaeval2025/diffusionxl_gen_images_resized"
os.makedirs(output_folder, exist_ok=True)

target_size = (460, 260)

def resize_and_convert(image_path, output_path, target_size):
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # 保证兼容性

        # 原始尺寸
        orig_w, orig_h = img.size
        target_w, target_h = target_size

        # 计算等比例缩放后尺寸
        ratio = min(target_w / orig_w, target_h / orig_h)
        new_size = (int(orig_w * ratio), int(orig_h * ratio))
        resized_img = img.resize(new_size, resample=Image.LANCZOS)

        # 创建白底画布
        canvas = Image.new("RGB", target_size, (255, 255, 255))

        # 计算居中位置
        paste_x = (target_w - new_size[0]) // 2
        paste_y = (target_h - new_size[1]) // 2

        # 粘贴
        canvas.paste(resized_img, (paste_x, paste_y))

        # 保存为 PNG
        canvas.save(output_path, format="PNG")

# ===== 批量处理 =====
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        output_name = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_folder, output_name)

        try:
            resize_and_convert(image_path, output_path, target_size)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print("✅ 所有图片已转换并保存为 PNG 格式，尺寸 460x260")