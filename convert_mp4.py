from moviepy.editor import ImageSequenceClip
import os
from PIL import Image
def images_to_video(image_folder, output_video, fps=20):
    # 获取所有图像文件
    image_files = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 过滤出可读取的图像文件
    valid_image_files = []
    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                valid_image_files.append(img_file)
        except Exception as e:
            print(f"无法读取图像文件 {img_file}: {e}")

    if not valid_image_files:
        print("没有可用的图像文件来创建视频。")
        return

    # 从有效的图像文件创建视频剪辑
    clip = ImageSequenceClip(valid_image_files, fps=fps)

    # 写入视频文件
    clip.write_videofile(output_video, codec='libx264')

images_to_video('./video_small/ckpt','./ckpt.mp4')