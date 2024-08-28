import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
from sklearn.linear_model import LinearRegression
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def read_image(image_path):
    """Reads and returns image."""
    return mpimg.imread(image_path)

def read_image_and_print_dims(image_path):
    """Reads and returns image.
    Helper function to examine how an image is represented.
    """
    #reading in an image
    image = mpimg.imread(image_path)
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    return image

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:  # if it is not gray-scale
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    # print("Hough lines: ", lines)
    line_img = np.zeros((*img.shape,3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def intersection_x(coef1, intercept1, coef2, intercept2):
    """Returns x-coordinate of intersection of two lines."""
    x = (intercept2 - intercept1) / (coef1 - coef2)
    return x

def draw_linear_regression_line(coef, intercept, intersection_x, img, imshape=[540, 960], color=[255, 0, 0],
                                thickness=2):
    # Get starting and ending points of regression line, ints.
    #print("Coef: ", coef, "Intercept: ", intercept, "intersection_x: ", intersection_x)
    point_one = (int(intersection_x), int(intersection_x * coef + intercept))
    if coef > 0:
        point_two = (imshape[1], int(imshape[1] * coef + intercept))
    elif coef < 0:
        point_two = (0, int(0 * coef + intercept))
    #print("Point one: ", point_one, "Point two: ", point_two)

    # Draw line using cv2.line
    cv2.line(img, point_one, point_two, color, thickness)
def find_line_fit(slope_intercept):
    """slope_intercept is an array [[slope, intercept], [slope, intercept]...]."""

    # Initialise arrays
    kept_slopes = []
    kept_intercepts = []
    #print("Slope & intercept: ", slope_intercept)
    if len(slope_intercept) == 1:
        return slope_intercept[0][0], slope_intercept[0][1]

    # Remove points with slope not within 1.5 standard deviations of the mean
    slopes = [pair[0] for pair in slope_intercept]
    mean_slope = np.mean(slopes)
    slope_std = np.std(slopes)
    for pair in slope_intercept:
        slope = pair[0]
        if slope - mean_slope < 1.5 * slope_std:
            kept_slopes.append(slope)
            kept_intercepts.append(pair[1])
    if not kept_slopes:
        kept_slopes = slopes
        kept_intercepts = [pair[1] for pair in slope_intercept]
    # Take estimate of slope, intercept to be the mean of remaining values
    slope = np.mean(kept_slopes)
    intercept = np.mean(kept_intercepts)
    #print("Slope: ", slope, "Intercept: ", intercept)
    return slope, intercept

def draw_lines(img, lines, color=[255, 0, 0], thickness=1):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_lines = []
    right_lines = []
    top_y = 1e6

    for line in lines:
        # no lane should be verticle view from the car
        for x1, y1, x2, y2 in line:
            if x1 != x2:
                slope = (y2 - y1) / (x2 - x1)
                if slope > 0:
                    left_lines.append([x1, y1, x2, y2])
                    # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                else:
                    right_lines.append([x1, y1, x2, y2])
                    # cv2.line(img, (x1, y1), (x2, y2), [0,0,255], thickness)
            if top_y > y1:
                top_y = y1
            if top_y > y2:
                top_y = y2

    # get the average position of each line
    if len(left_lines) > 0:
        left_line = [0, 0, 0, 0]
        for line in left_lines:
            assert (len(line) == 4)
            for i in range(4):
                left_line[i] += (line[i] / len(left_lines))
        slope = (left_line[3] - left_line[1]) / (left_line[2] - left_line[0])
        top_x = left_line[0] + (top_y - left_line[1]) / slope
        bottom_x = left_line[0] + (img.shape[0] - left_line[1]) / slope
        # cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color, thickness * 10)
        cv2.line(img, (int(bottom_x), img.shape[0]), (int(top_x), int(top_y)), color, thickness * 10)

    # get the average position of each line
    if len(right_lines) > 0:
        right_line = [0, 0, 0, 0]
        for line in right_lines:
            assert (len(line) == 4)
            for i in range(4):
                right_line[i] += (line[i] / len(right_lines))
        slope = (right_line[3] - right_line[1]) / (right_line[2] - right_line[0])
        top_x = right_line[0] + (top_y - right_line[1]) / slope
        bottom_x = right_line[0] + (img.shape[0] - right_line[1]) / slope
        # cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), [0,0,255], thickness * 10)
        cv2.line(img, (int(bottom_x), img.shape[0]), (int(top_x), int(top_y)), color, thickness * 10)

def find_linear_regression_line(points):
    # Separate points into X and y to fit LinearRegression model
    points_x = [[point[0]] for point in points]
    points_y = [point[1] for point in points]
    # points_x_print = [point[0] for point in points]
    # print("X points: ", points_x, "Length: ", len(points_x))
    # print("X points: ", points_x_print, "Length: ", len(points_x))
    # print("Y points: ", points_y, "Length: ", len(points_y))

    # Fit points to LinearRegression line
    clf = LinearRegression().fit(points_x, points_y)

    # Get parameters from line
    coef = clf.coef_[0]
    intercept = clf.intercept_
    #print("Coefficients: ", coef, "Intercept: ", intercept)
    return coef, intercept

def test_hough():
    img = read_image('test_images/solidYellowCurve2.jpg')
    img = grayscale(img)
    imgCanny = canny(img, 100, 200)
    imgHough = hough_lines(imgCanny, 1, np.pi/180, 200, 100, 10)
    tot = weighted_img(imgHough, img)
    plt.imshow(tot)
    plt.show()

def process_image(image):
    """Puts image through pipeline and returns 3-channel image for processing video below."""
    result = draw_lane_lines(image)
    #print(result.shape)
    return result

# Pipeline
def draw_lane_lines(image):
    """Draw lane lines in white on original image."""
    # Print image details
    # print("image.shape: ", image.shape)
    imshape = image.shape

    # Greyscale image
    greyscaled_image = grayscale(image)
    plt.subplot(2, 2, 1)
    plt.imshow(greyscaled_image, cmap="gray")

    # Gaussian Blur
    blurred_grey_image = gaussian_blur(greyscaled_image, 5)

    # Canny edge detection
    edges_image = canny(blurred_grey_image, 50, 150)

    # Mask edges image
    border = 0
    vertices = np.array([[(0, imshape[0]), (465, 320), (475, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    edges_image_with_mask = region_of_interest(edges_image, vertices)
    ## Plot masked edges image
    bw_edges_image_with_mask = cv2.cvtColor(edges_image_with_mask, cv2.COLOR_GRAY2BGR)
    plt.subplot(2, 2, 2)
    plt.imshow(bw_edges_image_with_mask)

    # Hough lines
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 45  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
    lines_image = hough_lines(edges_image_with_mask, rho, theta, threshold, min_line_len, max_line_gap)

    # Convert Hough from single channel to RGB to prep for weighted
    hough_rgb_image = lines_image
    # hough_rgb_image.dtype: uint8.  Shape: (540,960,3).
    # hough_rgb_image is like [[[0 0 0], [0 0 0],...] [[0 0 0], [0 0 0],...]]
    ## Plot Hough lines image
    plt.subplot(2, 2, 3)
    plt.imshow(hough_rgb_image, cmap='Greys_r')
    # Combine lines image with original image
    final_image = weighted_img(hough_rgb_image, image)
    ## Plot final image
    plt.subplot(2, 2, 4)
    plt.imshow(final_image, cmap='Greys_r')
    return final_image
