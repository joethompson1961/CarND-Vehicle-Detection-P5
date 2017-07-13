import cv2
import numpy as np
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import glob
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import validation
from analyze import inspect8

# Assumes input image color space is 'RGB', i.e. PNG file opened with matplotlib.image.imread().  
# It is also assumed that the input images are on a scale of 0-1 and they will be returned on that same scale
def convert_color(img, cspace = None):

    # apply color conversion if other than 'RGB'
    colorspace = None
    
    if cspace == 'HSV':
        colorspace = cv2.COLOR_RGB2HSV
    elif cspace == 'HLS':
        colorspace = cv2.COLOR_RGB2HLS
    elif cspace == 'LUV':
        colorspace = cv2.COLOR_RGB2LUV
    elif cspace == 'GRAY':
        colorspace = cv2.COLOR_RGB2GRAY
    elif cspace == 'YUV':
        colorspace = cv2.COLOR_RGB2YUV
    elif cspace == 'YCrCb':
        colorspace = cv2.COLOR_RGB2YCrCb

    if colorspace != None:
        img = cv2.cvtColor(img, colorspace)

    return img

# Compute color histogram features.
# Assumes input image color space is 'RGB', i.e. PNG file opened with matplotlib.image.imread().  
# Convert to a different color_space first using 'color_space' parameter
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH 
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, color_space=None, size=(32, 32)):
    # Convert image to new color space (if specified)
    
    img = convert_color(img, color_space)
    
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    
    return features

def color_histogram(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(img, spatial_feat=True, color_hist_feat=True, hog_feat=True,
                     cspace=None, spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256),
                     orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0, hog_vis=False, feature_vec=True):
    # Create a list to append feature vectors to
    features = []

    img = convert_color(img, cspace)

    # Apply bin_spatial() to get spatial color features
    if spatial_feat:
        spatial_features = bin_spatial(img, size=spatial_size)
        features.append(spatial_features)
    
    # Apply color_hist() to get color histogram features
    if color_hist_feat:
        hist_features = color_histogram(img, nbins=hist_bins, bins_range=hist_range)
        features.append(hist_features)
    
    if hog_feat:
        hog_features = []
        hog_images = []
        if hog_vis == False:
            if hog_channel == 'ALL':
                hog_features.append(get_hog_features(img[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=feature_vec))
                hog_features.append(get_hog_features(img[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=feature_vec))
                hog_features.append(get_hog_features(img[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=feature_vec))
                if feature_vec == True:
                    hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(img[:,:,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=feature_vec)
            features.append(hog_features)
        else:
            if hog_channel == 'ALL':
                hog_feat, hog_img = get_hog_features(img[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=feature_vec) 
                hog_features.append(hog_feat)
                hog_images.append(hog_img)
                hog_feat, hog_img = get_hog_features(img[:,:,1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=feature_vec) 
                hog_features.append(hog_feat)
                hog_images.append(hog_img)
                hog_feat, hog_img = get_hog_features(img[:,:,2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=feature_vec)
                hog_features.append(hog_feat)
                hog_images.append(hog_img)
                if feature_vec == True:
                    hog_features = np.ravel(hog_features)        
            else:
                hog_features, hog_img1 = get_hog_features(img[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=feature_vec)
            features.append(hog_features)

    # Return list of feature vectors
    if hog_vis == True:
        return features, hog_images
    else:
        return features
        
def train(spatial_size = (32,32), hist_bins = 32,
          hog_colorspace = 'YUV', hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channel="ALL", hog_vis=False ):
    # hog_colorspace Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    
    # Cars images
#    cars = glob.glob('C:/AMP/Udacity/condaenv/vdt_data/vehicles/KITTI_extracted/*.png')
    cars = glob.glob('C:/Udacity/SDCND/term1/resources/training-data/vehicle-detection_data/vehicles/My_Composite/*.png')

    # Not cars images
#    notcars = glob.glob('C:/AMP/Udacity/condaenv/vdt_data/non-vehicles/Extras/*.png')
    notcars = glob.glob('C:/Udacity/SDCND/term1/resources/training-data/vehicle-detection_data/non-vehicles/My_Composite/*.png')

    # Visualize example hog features
    if hog_vis == True:
        car_img = mpimg.imread(cars[0])
        car_img_luv = convert_color(car_img, 'LUV')
        feat, car_hogs = extract_features(car_img, spatial_feat=True, color_hist_feat=True, hog_feat=True,
                                          cspace=hog_colorspace, spatial_size = spatial_size, hist_bins = hist_bins,
                                          orient=hog_orient, pix_per_cell=hog_pix_per_cell,
                                          cell_per_block=hog_cell_per_block, hog_channel=hog_channel, hog_vis=True)
        notcar_img = mpimg.imread(notcars[14])
        notcar_img_luv = convert_color(notcar_img, 'LUV')
        feat, notcar_hogs = extract_features(notcar_img, spatial_feat=True, color_hist_feat=True, hog_feat=True,
                                             cspace=hog_colorspace, spatial_size = spatial_size, hist_bins = hist_bins,
                                             orient=hog_orient, pix_per_cell=hog_pix_per_cell,
                                             cell_per_block=hog_cell_per_block, hog_channel=hog_channel, hog_vis=True)
        inspect16(car_img, car_img_luv[:,:,0], car_img_luv[:,:,1], car_img_luv[:,:,2],
                  car_img, car_hogs[0], car_hogs[1], car_hogs[2],
                  notcar_img, notcar_img_luv[:,:,0], notcar_img_luv[:,:,1], notcar_img_luv[:,:,2],
                  notcar_img, notcar_hogs[0], notcar_hogs[1], notcar_hogs[2])
    
    shape = mpimg.imread(cars[0]).shape
    
    t=time.time()
    car_features = []
    for fn in cars:
        img = mpimg.imread(fn)
        
        if img.shape[2] == 4:
            img = img[:,:,0:3]
        if shape != img.shape:  # all images must be the same shape
            print("Shape mismatch", fn, "shape:", img.shape)
        else:
            # Extract color and HOG features
            feat = extract_features(img, spatial_feat=True, color_hist_feat=True, hog_feat=True,
                                    cspace=hog_colorspace, spatial_size = spatial_size, hist_bins = hist_bins,
                                    orient=hog_orient, pix_per_cell=hog_pix_per_cell,
                                    cell_per_block=hog_cell_per_block, hog_channel=hog_channel)
            try:
                validation.assert_all_finite(np.array(feat[2]).astype(np.float64))  # some images produce invalid hog results
                features = np.hstack((feat[0], feat[1], feat[2]))
                car_features.append(features)
            except:
#                print("oops:", fn)
                pass
    
    notcar_features = []
    for fn in notcars:
        img = mpimg.imread(fn)

        if img.shape[2] == 4:
            img = img[:,:,0:3]
        if shape != img.shape:
            print("Shape mismatch", fn, "shape:", img.shape)
        else:
            # Extract color and HOG features
            feat = extract_features(img, spatial_feat=True, color_hist_feat=True, hog_feat=True,
                                    cspace=hog_colorspace, spatial_size = spatial_size, hist_bins = hist_bins,
                                    orient=hog_orient, pix_per_cell=hog_pix_per_cell,
                                    cell_per_block=hog_cell_per_block, hog_channel=hog_channel)
            try:
                validation.assert_all_finite(np.array(feat[2]).astype(np.float64))  # some images produce invalid hog results
                features = np.hstack((feat[0], feat[1], feat[2]))
                notcar_features.append(features)
            except:
#                print("oops:", fn)
                pass
                           
#    t2 = time.time()
#    print(round(t2-t, 2), 'Seconds to extract HOG features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Fit a per-column scaler and use it to normalize X
    X_scaler = StandardScaler()
    X_scaler.fit(X)
    scaled_X = X_scaler.transform(X)  # normalize X
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    # # Try a Suppport Vector Classifier with linear kernel  
    # parameters = {'kernel':['linear'], 'C':[0.05, 0.1, 0.5, 1.0, 1.5, 5.0, 10.0]}
    # svc = SVC()
    # # Use a Suppport Vector Classifier with non-linear kernel to allow gamma exploration  
    # parameters = {'kernel':['rbf'], 'C':[0.05, 0.1, 0.5, 1.0, 1.5, 5.0, 10.0], 'gamma':[0.1, 1, 10]}
    # svr = SVC()
    
    # Use Linear Support Vector Classifier
    # parameters = {'C':[0.05, 0.1, 0.5, 1.0, 1.5, 5.0, 10.0]}
    # svr = LinearSVC()
    # svc = GridSearchCV(svr, parameters)
    svc = LinearSVC(C=0.05)
    
    # Check the training time for the SVC
    svc.fit(X_train, y_train)
#    print(svc.best_params_)
    
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
#     t=time.time()
#     n_predict = 50
#     print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
#     print('For these',n_predict, 'labels: ', y_test[0:n_predict])
#     t2 = time.time()
#     print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    return svc, X_scaler

if __name__ == "__main__":
    train()
