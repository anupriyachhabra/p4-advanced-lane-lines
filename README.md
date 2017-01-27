
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_cal/calibration1.jpg "Undistorted"
[image2]: ./output_images/distortion_correction/test1.jpg "Road Transformed"
[image3]: ./output_images/thresholded_binary/test1.jpg "Binary Thresholded Image"
[image4]: ./output_images/perspective_transform/test1.jpg "Warped Image"
[image5]: ./output_images/final_output/test1.jpg "Fit Visual"
[image6]: ./output_images/lane_lines_pixels/test1.jpg "Lane Line pixels "
[image7]: ./output_images/lane_lines_detected/test1.jpg "Lane Lines detected"
[video1]: ./project_video_output.mp4 "Fit Visual"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

###Camera Calibration

####1. Have the camera matrix and distortion coefficients been computed correctly and checked on one of the calibration images as a test?
The code for this step is contained in function caliberate_camera in lines 10 through 48 of the file called `find_lane_lines.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
I applied this distortion correction to the images in `camera_cal folder` using `cv2.undistort()` function and obtained this result:

![alt text][image1]

Above is just the output of first image , rest of the undistorted chessboard images are in folder [undistorted chessboard](./output_images/camera_cal)

###Pipeline (single images)

####1. Has the distortion correction been correctly applied to each image?
- I had saved the distortion coefficients and calibration matrix in a pickle file called `calibration_data.p` during the camera calibration step.
- In my pipeline I load the calibration matric and distortion coefficients from pickle file and use the `cv2.undistort()` function to undistort images
- To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Above is just the output of first test image , rest of the undistorted test images are in folder [undistorted test images](./output_images/distortion_correction)


####2. Has a binary image been created using color transforms, gradients or other methods?
- After correcting the image for distortion I have created a region of interest by masking the extra area around the image.
- Masking the area made it easy for me to apply and decide which gradient and color thresholds to use.
- After that I experimented with various color thresholding and gradient thresholding techniques.
- I found that changing the image to hls color space made it easier to apply the thresholds.
- After converting the image to hls color space I used the s channel to apply sobelx gradient
- After experimenting with various number I came up with min threshold 10 and max 100 for sobelx
- For s channel I also used color thresholding with min threshold 190 and  max threshold 230
- Finally I combined the output of the two thresholded images using bitwise or and following is an example output image

![alt text][image3]

Above is just the output of first test image , rest of the thresholded binary images are in folder [thresholded binary images](./output_images/thresholded_binary)


####3. Has a perspective transform been applied to rectify the image?

The code for my perspective transform includes a function called `warper()`, which appears in lines 252 through 272 in the file `find_lane_lines.py`:
The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.
I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```

- I have applied perspective transform on the thresholded binary images as this will be useful in the next step while calculating the radius of curvature.
- First thresholded image with perspective tranform applied is as following
![alt text][image4]

Above is just the output of first test image , rest of the perspective transformed are in folder [warped images](./output_images/perspective_transform)


####4. Have lane line pixels been identified in the rectified image and fit with a polynomial?
- I have identified lane pixels using a histogram of hot pixels and used a sliding window approach to find the best fit around the previously found pixels.
- Following is an image with detected lane pixels for left and right lanes

![alt text][image6]
Above is just the output of first test image , rest of the images with lane pixels detected are in folder [warped images](./output_images/lane_lines_pixels)

- Next I have found the min and max points from left and right lines and combined them using to draw an area of lane lines as follows
![alt text][image6]
Above is just the output of first test image , rest of the images with lane area drawn are in folder [warped images](./output_images/lane_lines_detected)


####5. Having identified the lane lines, has the radius of curvature of the road been estimated? And the position of the vehicle with respect to center in the lane?
- Yes I have used the techniques used in Lecture 34 , Measuring Curvature to find the lane curvature and plot it on the image.
- Following is an image with lane curvature marked. This is also the final step of the pipeline

![alt text][image5]

Above is just the output of first test image , rest of the images with lane area drawn are in folder [warped images](./output_images/final_output)


###Pipeline (video)

####1. Does the pipeline established with the test images work to process the video?

Yes, here's a [link to my video result](./project_video_output.mp4)

---

###README

####1. Has a README file been included that describes in detail the steps taken to construct the pipeline, techniques used, areas where improvements could be made?

Yes, this is the file.


---
##Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

- I could not detect anything useful from sobelx or sobely or magnitude gradient initially when I was using rgb color space.
- Converting to hls space made the work a lot easier for me
- I also faced some issues with deciding the source and destination points for perspective transform.
- Then I used the source and destination points from the example write up and they worked for me.
- Initially I had not used a region of interest mask, but when I was plotting a histogram of hot pixels I was getting lot of outlier points.
- So I put in a region of interest, initially I only put it before finding lane pixels, but then I put it in the beginning of pipeline and I found that this improved
my results a lot.
- As I have used some hardcoded points, my pipeline will fail for images with different width and height to the current test images and pipeline.
- If possible I would like to make the pipeline more robust for any image size.


