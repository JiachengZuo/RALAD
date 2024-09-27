import os

# 假设有一个文件夹路径，我们将其命名为"image_folder"
image_folder = "./fusion_dataset/new_carla/image_2"

# 获取该文件夹下所有png文件的名称（去掉.png后缀）
png_files = [file[:-4] for file in os.listdir(image_folder) if file.endswith('.png')]

# 将这些文件名保存到txt文件中
output_file = "./output_file.txt"
with open(output_file, 'w') as f:
    for file_name in png_files:
        f.write(file_name + '\n')

# 返回生成的文件路径以供参考
output_file
