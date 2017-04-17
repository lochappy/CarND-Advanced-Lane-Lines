
## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./output_images/image_correction.png "Undistorted"
[image2]: ./output_images/image_correction_samples.png "Road Transformed"
[image3]: ./output_images/thresholded_images.png "Binary Example"
[image3a]: ./output_images/mask_image.png "Mask"
[image3b]: ./output_images/clean_thresholded_images.png "Clean Binary Example"
[image4]: ./output_images/birdeye_view_images.png "Warp Example"
[image5]: ./output_images/fitted_lanelines.png "Fit Visual"
[image6]: ./output_images/marked_lane.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 5th code cell of the IPython notebook located in "./AdvancedLaneLines.ipynb". I broke this step to 2 small ultility functions:

* `calibrate_camera()`: This function calibrates the camera given the list of chessboard images and the grid dimensions of the chessboard. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

* `undistort_image()`: This function corrects the distortion of the given image, camera matrix and disortion coefficients. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
The left column is the raw images, and the right column is the corrected one
![alt text][image2]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. This step is perform in `colorAndGradientThreshold()` function of the 8th code cell of the IPython notebook located in "./AdvancedLaneLines.ipynb". Specifically, this process walks through following procedure:
* Blur the input image `img` by a Gaussian filter, results in `blurred_img`.
* The `blurred_img` is then converted to gray image `gray`.
* Gradient along x (`gradx`) and y (`grady`) are computed.
* Magnitude and direction threshold are computed based on `gradx` and `grady`, results in `mag_binary` and `dir_binary`.
* The `blurred_img` is then converted to HLS image `hls`.
* Color threshold is then applied on the `s` channel of the `hls`.
* Finally, a OR combination of `mag_binary`, `dir_binary` and `s` produces the final ouput binary image `binary`

Here's an example of my output for this step.
![alt text][image3]

In other to refine the thresholded images, I applied a mask, which defined my region of interest, to the thresholded image.
![alt text][image3a]

The final result looks a lot cleaner as shown in the following figure:
![alt text][image3b]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_forward()` and `warp_backward()`, which appears in the 10th code cell of the IPython notebook. I chose the hardcode the source and destination points in the following manner:
```
#define the source coordinates
left_bottom = (220, 720)
right_bottom = (1280 - left_bottom[0], 720)
apex1 = (585, 460)
apex2 = (700, apex1[1])
birdeye_roi_src = np.array([[left_bottom, apex1, apex2, right_bottom,]], dtype=np.float32)

#define the destination coordinates
left_bottom = (300, 720)
right_bottom = (1280 - left_bottom[0], 720)
left_top = (left_bottom[0], 0)
right_top = (right_bottom[0], 0)
birdeye_roi_dst = np.array([[left_bottom, left_top, right_top, right_bottom,]], dtype=np.float32)

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 220, 720      | 300, 720      | 
| 585, 460      | 300, 0        |
| 700, 460      | 980, 0        |
| 1060, 720     | 980, 720      |

With the 2 predefined sets of point, I calculated the 2 transforming matrices as ```perspectiveTransformMatrix```, which tranforms the source to the destination, and ```invPerspectiveTransformMatrix```, which tranforms the destination to the source.

The `warp_forward()` and `warp_backward()` function warp the input image (`inputImg`) from the camera viewing angle to the birdeye one and vice versa, respectively. 

I verified that my perspective transform was working as expected by drawing the `birdeye_roi_src` points (in red) onto a test images, and warped them to the birdeye view to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I did this in the 13th code cell of the IPython notebook in the function `fitLaneLine()`. In general, this function  takes a binary birdeye view image (`binary_warped`) as an input and processes through following steps to find the lanelines:
* Identifying the bases (`leftx_base` and `rightx_base`) of the left and right lane line by finding the left and right peaks of the histogram taken along all the columns in the lower half of the `binary_warped`.
* The `binary_warped` is horizontally sliced into `nwindows` pieces. These slices will be scanned from the bottom to the top. The left and right peaks of the histogram along the columns of the current scanned slice will be identified within a `margin` offset to the one of the previous slice. The bottom most slice takes the `leftx_base` and `rightx_base` as its references. If the left or right peak of the current slice cannot be found, it will be assigned by the one of the previous slice.
* Finally, two 2nd order polynomials are fitted across the `nwindows` left and right peaks.

The output of this step is presented in the following figure
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the 14th code cell of the IPython notebook in the function `compute_curvature()` and `compute_position()`:

* `compute_curvature()` computes the radius of curvature of the laneline by taking `line`, which is a `Line()` object class defined in the 12th code cell of the IPython notebook, as an input. Then it converts the unit of `line.recent_xyfitted` from pixels to meters by multiply them with an appropriate scale factors (`xm_per_pix` and `ym_per_pix`), results in `xm` and `ym`. Next, a 2nd order poly line `fit_cr` is fitted on the pair `(ym,xm)`. Finally, the curvature is computed by taking the average of all curvatures of `fit_cr` at all `ym`.

* `compute_position()` calculates the position of the vehicle with respect to the center of the lane by taking the `leftLaneline` and `rightLaneLine`, which are also `Line()` object class, as inputs. Then it computes the lane center in pixels (`lane_center_x`) based on the horizontal position of left and right laneline. Next, assuming that the car center is horizontally aligned with the image center at `640.` pixel, the function compute the offset distance of the car center with respect to the lane center in pixels (`center_offset_in_pixels`). Finally the `center_offset_in_pixels` is converted to meter space `center_offset_in_meters` as the final returned result..

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the 16th code cell of the IPython notebook in the function `drawRegionOfTheLane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/lochappy/CarND-Advanced-Lane-Lines/blob/master/detected_laneline_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It took me more than 30 hours to complete the project mostly struggling in seeking appropriate parameters for image thresholding, as well as the warping function.

I have also tried out the convolution method for detecting laneline pixels, and spent some time in tuning its parameters. However, it turned out that the histogram method was a better solution.

The pipeline will likely fail in case of harsh weather condition, like snowing, raining, etc..., or in case the laneline has special shape that the 2nd order poly line cannot fit in.

Laneline pixel detection is the most vulnerable step of my pipeline. Saying that the current pipeline uses hardcoded threshold values, which are very sesitive to the image condition, to detect the laneline pixels. Therefore, one way to make the pipeline more robust is to improve robustness the laneline pixel detection step by using convolutional neural network, for instance.

