# Object-Detection

# Vehicle Detection


The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output_images/car_samples.png "Car Samples"
[image3]: ./output_images/non_car_samples.png "Non-car Samples"
[image4]: ./output_images/HOG_comparison.png "HOG Comparison"
[image5]: ./output_images/bbox_vis.png "Region Proposal Visualisation"
[image6]: ./output_images/heatmap.png "Heatmap"
[image7]: ./output_images/video_thumbnail.jpg "Video Thumbnail"


## Basic Summary of Dataset
We begin in Vehicle_Detector.ipynb , with a summary of the provided dataset:

No. of car images:  8792

No of non-car images:  8968

Image shape:  (64, 64, 3)

The dataset is fairly balanced.

### Car Image Examples
![alt_text][image1]

### Non-car Image Examples
![alt_text][image3]

## Feature Extraction Method

`bin_spatial()` computes binned colour features by scaling images down.

`color_hist()` computes colour histogram features.

`get_hog_features()` returns HOG features and visualisation.

`extract_features()` wraps and combines the above functions.

## HOG Visualisation
![alt_text][image4]

After experimenting different color spaces and using different channels, I choose to use all the YCrCb color channels as it resulted in the highest accuracy in the classifier .

One might be tempted to tune the parameters optimising purely for accuracy. However, I had to consider the computational complexity of the chosen parameters. More pixels per cell and more cells per block could result in tighter boxes, but also in higher computational complexity. The resulting parameters chosen were found by trial annd error, and appear to be a cursory sweetspot given the processor on my laptop.

## Data Preparation
I felt the dataset was large enough, and did not require further augmentation for this relatively simple classifier.
we apply a scaler to the feature vector, and split the data into training/validation sets.

## Classifier
I run the classifier on the dataset with a 0.989 accuracy. There is a chance of overfitting here,
but its effects proved to be acceptable for the purposes of this exercise.

## Sliding Window Search
One can find the HOG window search routine. This is a region proposer that scans select portions of the image
and checks for presence of a vehicle. The cell size and cell position are determined by the way the road is laid out, and the
distance of vehicles from the camera. 

An example is shown below, where the window search is told to look. Smaller search area farther away is given smaller cells to search within, given that cars appear smaller further away.

![alt_text][image5]

It is immediately apparent that the search area includes portions unlikely to contain vehicles (like left of the highway barrier). Regardless, it makes sense to include this search region as the driver could choose to drive in the middle lane instead.
 
## Temporal Heatmapping
I introduced a heatmap of successful hits in the frame. The outputs of this heatmap are recorded and added to a queue of length n in `detect_history()` . 

By using this queue, I'm able to smooth my bounding boxes between frames. This step helps reduce false positives that would be mis-classified over consecutive frames. The result is improved accuracy and a smoother overall result.

Below is the visualization of the heatmap next to the resulting bounding boxes.

![alt_text][image6]

## Final Pipeline
All that's left to do now is to process the entire video


 

