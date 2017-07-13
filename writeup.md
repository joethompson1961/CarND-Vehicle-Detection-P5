##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring
* detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_windows2.png
[image5]: ./output_images/boxes_heat_labels_output.png
[image6]: ./output_images/labels_6.png
[image7]: ./output_images/project_video_output_6.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `train()` function at lines 176 through 219 of the file called `train.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(12,12)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried using various combinations of parameters to train a Linear SVM classifier and initially chose the parameters color space=`LUV`, `orientations=9`, `pixels_per_cell=(12,12)`, `cells_per_block=(2, 2)`.  These parameters were selected for high training test accuracy as well as better compute performance, i.e. lower CPU overhead with larger cell size.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the `train()` function at lines 225 through 254 of the file called `train.py`.

I trained a linear SVM using HOG and color features extracted from the training dataset images. I used `sklearn.GridSearchCV()` to explore the values `[0.05, 0.1, 0.5, 1.0, 1.5, 5.0, 10.0]` for SVM `C` parameter and selected `C=0.05` based on the result in `svc.best_params_`.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window is implemented in the `find_cars()` fuction at lines 108 through 171 of the file called `vdt.py`.
 
For better compute performance I had decided to do HOG computations only once per video frame and then extract the hog features for each window section from the pre-computed HOG features.  Because of this it was necessary to use a sliding window algorithm that stepped the sliding window by a multiple of HOG cell widths, e.g. with a hog cell width of 8 a window step of 2 cells would be 16 pixels. This was necessary to ensure the window section exactly aligns with the pre-computed hog features.

I also decided to search using different scales tailored to detect cars at different distances. I chose the scales based on an approximation of how the size of vehicles appearing at different distances in the image would match the typical vehicle sizes in the training images.

In addition I limited the window search for the scaled searches to regions of interest that made sense for the chosen scale. For example, there is no point in searching for distant cars at the bottom of the image.

This picture shows the regions of interest used for the selected scales. 

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.

To get a good final outcome I found it necessary to augment my training dataset with images extracted from the project video using my specific windowing scheme. Perhaps SVM is not able to generalize as well other classification methods such as CNN? In any case, with the augmented dataset it did finally provided a nice result.  Here are some example images:

![alt text][image4] 

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](C:/AMP/Udacity/condaenv/CarND-Vehicle-Detection/output_video.mp4)

Here's a [link to my video result](./output_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded a 10-frame running history of the positions of positive detections, i.e. the heatmaps. For each frame I summed the heatmap history to combine the overlapping detections and then applied a threshold to filter out false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

This is implemented in the `process_image()` function at lines 286 through 295 in the file `vdt.py'.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a demonstration of the entire pipeline flow for 6 sequential frames: output from the sliding window classifier, their corresponding heatmaps, the integrated heatmap accumulated over all six frames (with thresholds applied to filter false positives), the corresponding output of `scipy.ndimage.measurements.label()` on thresholded heatmaps, and the resulting output bounding boxes:

![alt text][image5]

### Here is the full view of the output `scipy.ndimage.measurements.label()` on the integrated heatmap at the last frame in the series:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Besides all the usual debugging issues, The most serious problem was getting a good result from the classifier. In the end the fix that worked was to augment the training dataset with images extracted from the project video using my specific scaling/windowing configuration. I expect it's unlikely this solution will work well for any other road except the one in the project video.  To improve I would first seek out a better classifier, a CNN perhaps, and then focus on creating a more robust and larger training set.

After that I think a lot more experimentation and tuning is required to create a system that performs robustly in varied environments.

There was also an annoying problem with some of the training data that caused NaN exceptions in StandardScaler.transform().  I haven't figured out the root cause but was able to work around it buy catching exceptions using the validation() function priort to transform().

