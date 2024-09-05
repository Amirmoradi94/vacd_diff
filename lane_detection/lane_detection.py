
from moviepy.editor import VideoFileClip
from utils import *


left_fit_hist = np.array([])
right_fit_hist = np.array([])

"""
video_output = 'samples/project_video_output.mp4'
clip1 = VideoFileClip("samples/project_video.mp4")
output_clip = clip1.fl_image(lambda frame: lane_finding_pipeline(frame, left_fit_hist, right_fit_hist))
output_clip.write_videofile(video_output, audio=False)
"""
"""

test_image = cv2.imread('samples/test5.jpg')
processed_image = lane_finding_pipeline(test_image, left_fit_hist, right_fit_hist)
cv2.imwrite('samples/test_image_output.jpg', processed_image)
"""

test_image = cv2.imread('samples/test5.jpg')
processed_image, diffrence_angle = process_image(test_image)
print(diffrence_angle)
cv2.imwrite('samples/test_image_output.jpg', processed_image)