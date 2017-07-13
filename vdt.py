import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from train import train, convert_color, bin_spatial, color_histogram, get_hog_features
from analyze import inspect1, inspect3

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from collections import deque
import copy
from sklearn.utils import validation

def detect_vehicles_video(img):
    current_heatmap = add_heat(img, hot_windows)
    heatmaps.append(current_heatmap)
    heatmap_sum = sum(heatmaps)

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, boxes, color=(0, 0, 1), thick=6):
    # Make a copy of the image
#    imcopy = np.copy(img)

    # Iterate through the bounding boxes
    for box in boxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img, box[0], box[1], color, thick)
    # Return the image copy with boxes drawn
    return img

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    img_shp = img.shape
    
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shp[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shp[0]
    
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a single function that can extract features using hog sub-sampling and make predictions
attempts = 0
save_sections = False
def find_cars(img, xstart, xstop, ystart, ystop, scale,
              svc, X_scaler,
              colorspace, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins):
    global attempts
    attempts += 1

    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, colorspace)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Define blocks and steps as above
#     nxblocks = (ch1.shape[1] // pix_per_cell) - 1
#     nyblocks = (ch1.shape[0] // pix_per_cell) - 1 
    nxblocks = (ch1.shape[1] // pix_per_cell)
    nyblocks = (ch1.shape[0] // pix_per_cell) 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    train_window = 64
    hog_size = (train_window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
#     nxsteps = ((nxblocks - hog_size) // cells_per_step)
#     nysteps = ((nyblocks - hog_size) // cells_per_step)
    nxsteps = ((nxblocks - hog_size) // cells_per_step) + 1
    nysteps = ((nyblocks - hog_size) // cells_per_step) + 1

    boxes = []
    section_no = 0
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for the section
            hog_features = []
            hog_features.append(hog1[ypos:ypos+hog_size, xpos:xpos+hog_size,:,:,:])
            hog_features.append(hog2[ypos:ypos+hog_size, xpos:xpos+hog_size,:,:,:])
            hog_features.append(hog3[ypos:ypos+hog_size, xpos:xpos+hog_size,:,:,:])
            hog_features = np.ravel(hog_features)

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+train_window, xleft:xleft+train_window], (64,64))
            if save_sections:
                if subimg.shape[2] == 4:
                    subimg = subimg[:,:,0:3]
                fn = './sections/' + str(attempts) + '_' + str(section_no) + '.png'
                mpimg.imsave(fn, subimg)
                section_no += 1
          
            # Extrace color features for the section
            spatial_features = bin_spatial(subimg, size=spatial_size).ravel()
            hist_features = color_histogram(subimg, nbins=hist_bins)

            test_features = np.hstack((spatial_features, hist_features, hog_features))
            
            try:
                validation.assert_all_finite(np.array(hog_features).astype(np.float64))
                # Scale features and make a prediction
                test_features = X_scaler.transform(test_features)    
                prediction = svc.predict(test_features)
                
                if prediction == True:
                    xleft_draw = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(train_window*scale)
                    a = xleft_draw + xstart
                    b = ytop_draw  + ystart
                    c = xleft_draw + win_draw + xstart
                    d = ytop_draw  + win_draw + ystart
                    boxes.append( ((a,b),(c,d)) )
            except:
                pass
            
    return boxes

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        color = (0,0,1)
        cv2.rectangle(img, bbox[0], bbox[1], color, 5)
    # Return the image
    return img

save_frames = False
frame_num = 0
heatmaps = deque(maxlen=10)
def process_image(img):
    global save_frames
    global frame_num
    global heatmaps
    global classifier
    global X_scaler
    global orient
    global pix_per_cell
    global cell_per_block
    global spatial_size
    global hist_bins

    if np.mean(img) > 1:
        img = img.astype(np.float32)/255
    
    # If video image contains 4 color channels per frame then reduce it to 3 (what is the 4th channel?)
    if img.shape[2] == 4:
        img = img[:,:,0:3]

    dst = np.copy(img)

    # Save each frame of the original video
    if save_frames:
        fn = './project_video/project_video_' + str(frame_num) + '.png'
        mpimg.imsave(fn, dst)

    all_boxes = []

    # Search for far away cars
    xstart = 0
    xstop = 1280
    ystart = 400
    ystop = 496
    scale = 1.0
    aboxes = find_cars(img, xstart, xstop, ystart, ystop, scale,
                       classifier, X_scaler,
                       colorspace, orient, pix_per_cell, cell_per_block,
                       spatial_size, hist_bins)
    all_boxes = all_boxes + aboxes

    # Search intermediat distance cars
    xstart = 0
    xstop = 1280
    ystart = 380
    ystop = 600
    scale = 1.75
    bboxes = find_cars(img, xstart, xstop, ystart, ystop, scale,
                       classifier, X_scaler,
                       colorspace, orient, pix_per_cell, cell_per_block,
                       spatial_size, hist_bins)
    all_boxes = all_boxes + bboxes
     
    # Search for close cars
    xstart = 0
    xstop = 1280
    ystart = 360
    ystop = 684
    scale = 2.5
    cboxes = find_cars(img, xstart, xstop, ystart, ystop, scale,
                       classifier, X_scaler,
                       colorspace, orient, pix_per_cell, cell_per_block,
                       spatial_size, hist_bins)
    all_boxes = all_boxes + cboxes

#     draw_boxes(dst, cboxes, color=(1, 0, 0), thick=6)
#     draw_boxes(dst, bboxes, color=(0, 1, 0), thick=6)
#     draw_boxes(dst, aboxes, color=(0, 0, 1), thick=6)
    
#     f, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
#     f.tight_layout()
#     ax1.imshow(dst, cmap='gray')
#     ax1.set_title('Near (Red), Med (Green) and Far (Blue) Sliding Window Regions', fontsize=10)
#     plt.subplots_adjust(left=0., right=1., top=0.9, bottom=0.)
#     plt.show()

    # Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
    heatmap = np.zeros_like(dst[:,:,0]).astype(np.float)
    heatmap = add_heat(heatmap, all_boxes)
     
    heatmaps.append(heatmap)
    heatmap_sum = sum(heatmaps)
    heatmap_sum = apply_threshold(heatmap_sum, 10)
    labels = label(heatmap_sum)
#     plt.imshow(labels[0], cmap='gray')
#     plt.show()
    draw_labeled_bboxes(dst, labels)    
        
    # Add useful info, e.g. frame number to output frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg = 'frame: ' + str(int(frame_num))
    color = (0,0,1)
    cv2.putText(dst,msg,(50,50), font, 0.85, color, 2)

    # Save each frame of augmented video with vehicle bounding boxes
    if save_frames:
        fn = './temp/project_video_windows_' + str(frame_num) + '.png'
        mpimg.imsave(fn, dst)
        fn = './temp/project_video_heatmap' + str(frame_num) + '.png'
        mpimg.imsave(fn, heatmap)
        fn = './temp/project_video_heatmap_thresh' + str(frame_num) + '.png'
        mpimg.imsave(fn, heatmap_sum)

    frame_num += 1

    dst = dst*255
    dst = dst.astype(int)
    
    return dst

# hog feature extraction parameters
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 12
cell_per_block = 2
channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (64, 64)
hist_bins = 16
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load saved classifier from file.')
    parser.add_argument(
        'preload',
        type=str,
        default='',
        nargs='?',
        help='Load saved classifier from file instead of training from scratch.'
    )
    args = parser.parse_args()

    # User option to load saved classifier using "preload" parameter 
    if args.preload != '':
        classifier = joblib.load('./classifier.pkl')
        X_scaler = joblib.load('./scaler.pkl')
    else:
        classifier, X_scaler = train(spatial_size = spatial_size, hist_bins = hist_bins,
                                     hog_colorspace = colorspace,
                                     hog_orient = orient,
                                     hog_pix_per_cell = pix_per_cell,
                                     hog_cell_per_block = cell_per_block,
                                     hog_channel = channel,
                                     hog_vis=False)

        # Save the classifier to file
        joblib.dump(classifier, './classifier.pkl', compress=9)
        joblib.dump(X_scaler, './scaler.pkl', compress=9)
    
#     # process test images
# #    imagesPath = './test_images/test1*.jpg'
#     imagesPath = './test_images/project_video_*.jpg'
#     test_images = glob.glob(imagesPath)
#     for filename in test_images:
#         src = mpimg.imread(filename)
#         dst = process_image(src)
#         if save_sections == False:  # don't save output frames when saving section frames
#             fn = './temp/project_video_output_' + str(frame_num-1) + '.png'
#             dst = dst.astype(np.float32)/255
#             mpimg.imsave(fn, dst)
#         print("...", frame_num, filename)
    
    # process the video
    clip1 = VideoFileClip("project_video.mp4")
#    clip1 = VideoFileClip("project_video.mp4")
    processed_video = 'output_video.mp4'
    processed_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    processed_clip.write_videofile(processed_video, audio=False)

    exit()