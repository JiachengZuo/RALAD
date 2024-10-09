from moviepy.editor import VideoFileClip

# MP4文件路径
mp4_path = './images/视频2.mp4'
# 输出GIF的路径
gif_path = './images/2.gif'

# 加载视频文件
clip = VideoFileClip(mp4_path)

# 将视频转换为GIF，这里可以设置fps参数来调整GIF的帧率
clip.write_gif(gif_path, fps=30)

# 释放资源
clip.close()
