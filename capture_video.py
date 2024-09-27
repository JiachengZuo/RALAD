import cv2

# 视频文件路径
video_path = './DJI_20240904160115_0151_D.mp4'

# 读取视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 计算每0.5秒对应的帧数
frame_interval = int(fps * 0.5)

# 初始化帧计数器
frame_count = 0
count = 0
# 循环遍历视频中的每一帧
while True:
    # 读取下一帧
    ret, frame = cap.read()

    # 如果读取帧失败，则退出循环
    if not ret:
        break

    # 每隔frame_interval帧保存一张图片
    if frame_count % frame_interval == 0:
        # 图片文件名，可以根据需要自定义命名规则
        image_filename = f'image_{count}.png'
        # 保存图片
        cv2.imwrite(image_filename, frame)
        print(f'Saved {image_filename}')
        count += 1


    # 更新帧计数器
    frame_count += 1

# 释放视频捕获对象
cap.release()
