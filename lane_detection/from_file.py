from laneDetectionMethods import *

#Test on a video1: straight lane
white_output = 'result_straight.mp4'
clip1 = VideoFileClip("lane_detection/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

#Test on a video2: curved lane (you'll notice that this implementation can only detect straight lanes)
white_output = 'result_curve.mp4'
clip1 = VideoFileClip("lane_detection/challenge.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)