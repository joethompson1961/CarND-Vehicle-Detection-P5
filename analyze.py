import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import glob

def inspect1(src, dst):
    """
    Output 2 images for comparison.
    
    These can be any two images but typically the first is the original and the 2nd is processed.
    """    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    f.tight_layout()
    ax1.imshow(src, cmap='gray')
    ax1.set_title('Vehicle', fontsize=10)
    ax2.imshow(dst, cmap='gray')
    ax2.set_title('Not Vehicle', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
def inspect3(src, dst1, dst2, dst3):
    """
    Output 4 images for comparison.
    
    These can be any four images but typically the first is the original and the rest are processed.
    """    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 5))
    f.tight_layout()
    ax1.imshow(src)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(dst1, cmap='gray')
    ax2.set_title('Channel 0', fontsize=10)
    ax3.imshow(dst2, cmap='gray')
    ax3.set_title('Channel 1', fontsize=10)
    ax4.imshow(dst3, cmap='gray')
    ax4.set_title('Channel 2', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def inspect2x2(img1, img2, img3, img4):
    """
    Output 4 images for comparison.
    
    These can be any four images but typically the first is the original and the rest are processed.
    """    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title('Example 1', fontsize=10)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('Example 2', fontsize=10)
    ax3.imshow(img3, cmap='gray')
    ax3.set_title('Example 3', fontsize=10)
    ax4.imshow(img4, cmap='gray')
    ax4.set_title('Example 4', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def inspect8(title, img1, img2, img3, img4, img5, img6, img7, img8):
    """
    Output 4 images for comparison.
    
    These can be any four images but typically the first is the original and the rest are processed.
    """    
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title, fontsize=10)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('LUV Channel 0', fontsize=10)
    ax3.imshow(img3, cmap='gray')
    ax3.set_title('LUV Channel 1', fontsize=10)
    ax4.imshow(img4, cmap='gray')
    ax4.set_title('LUV Channel 2', fontsize=10)
    ax5.imshow(img5)
    ax5.set_title('', fontsize=10)
    ax6.imshow(img6, cmap='gray')
    ax6.set_title('HOG Channel 0', fontsize=10)
    ax7.imshow(img7, cmap='gray')
    ax7.set_title('HOG Channel 1', fontsize=10)
    ax8.imshow(img8, cmap='gray')
    ax8.set_title('HOG Channel 2', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def inspect16(img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16):
    """
    Output 4 images for comparison.
    
    These can be any four images but typically the first is the original and the rest are processed.
    """    
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, figsize=(8, 8))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title('Car', fontsize=10)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('Car LUV CH0', fontsize=10)
    ax3.imshow(img3, cmap='gray')
    ax3.set_title('Car LUV CH1', fontsize=10)
    ax4.imshow(img4, cmap='gray')
    ax4.set_title('Car LUV CH2', fontsize=10)
    ax5.imshow(img5)
    ax5.set_title('Car', fontsize=10)
    ax6.imshow(img6, cmap='gray')
    ax6.set_title('Car HOG CH0', fontsize=10)
    ax7.imshow(img7, cmap='gray')
    ax7.set_title('Car HOG CH1', fontsize=10)
    ax8.imshow(img8, cmap='gray')
    ax8.set_title('Car HOG CH2', fontsize=10)
    ax9.imshow(img9)
    ax9.set_title('Not Car', fontsize=10)
    ax10.imshow(img10, cmap='gray')
    ax10.set_title('Not Car LUV CH0', fontsize=10)
    ax11.imshow(img11, cmap='gray')
    ax11.set_title('Not Car LUV CH1', fontsize=10)
    ax12.imshow(img12, cmap='gray')
    ax12.set_title('Not Car LUV CH2', fontsize=10)
    ax13.imshow(img13)
    ax13.set_title('Not Car', fontsize=10)
    ax14.imshow(img14, cmap='gray')
    ax14.set_title('Not Car HOG CH0', fontsize=10)
    ax15.imshow(img15, cmap='gray')
    ax15.set_title('Not Car HOG CH1', fontsize=10)
    ax16.imshow(img16, cmap='gray')
    ax16.set_title('Not Car HOG CH2', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def inspect5x6(img1, img2, img3, img4, img5, img6, img7, img8, img9, img10,
               img11,img12, img13, img14, img15, img16, img17, img18, img19, img20,
               img21, img22, img23, img24, img25, img26, img27, img28, img29, img30):
    f, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15), (ax16, ax17, ax18, ax19, ax20), (ax21, ax22, ax23, ax24, ax25), (ax26, ax27, ax28, ax29, ax30)) = plt.subplots(6, 5, figsize=(12, 9))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    ax1.set_title('frame 1', fontsize=10)
    ax6.imshow(img2, cmap='gray')
    ax6.set_title('frame 2', fontsize=10)
    ax11.imshow(img3, cmap='gray')
    ax11.set_title('frame 3', fontsize=10)
    ax16.imshow(img4, cmap='gray')
    ax16.set_title('frame 4', fontsize=10)
    ax21.imshow(img5, cmap='gray')
    ax21.set_title('frame 5', fontsize=10)
    ax26.imshow(img6, cmap='gray')
    ax26.set_title('frame 6', fontsize=10)
    ax2.imshow(img7, cmap='gray')
    ax2.set_title('heatmap 1', fontsize=10)
    ax7.imshow(img8, cmap='gray')
    ax7.set_title('heatmap 2', fontsize=10)
    ax12.imshow(img9, cmap='gray')
    ax12.set_title('heatmap 3', fontsize=10)
    ax17.imshow(img10, cmap='gray')
    ax17.set_title('heatmap 4', fontsize=10)
    ax22.imshow(img11, cmap='gray')
    ax22.set_title('heatmap 5', fontsize=10)
    ax27.imshow(img12, cmap='gray')
    ax27.set_title('heatmap 6', fontsize=10)
    ax3.imshow(img13, cmap='gray')
    ax3.set_title('thresholded 1', fontsize=10)
    ax8.imshow(img14, cmap='gray')
    ax8.set_title('thresholded 2', fontsize=10)
    ax13.imshow(img15, cmap='gray')
    ax13.set_title('thresholded 3', fontsize=10)
    ax18.imshow(img16, cmap='gray')
    ax18.set_title('thresholded 4', fontsize=10)
    ax23.imshow(img17, cmap='gray')
    ax23.set_title('thresholded 5', fontsize=10)
    ax28.imshow(img18, cmap='gray')
    ax28.set_title('thresholded 6', fontsize=10)
    ax4.imshow(img19, cmap='gray')
    ax4.set_title('labels 1', fontsize=10)
    ax9.imshow(img20, cmap='gray')
    ax9.set_title('labels 2', fontsize=10)
    ax14.imshow(img21, cmap='gray')
    ax14.set_title('labels 3', fontsize=10)
    ax19.imshow(img22, cmap='gray')
    ax19.set_title('labels 4', fontsize=10)
    ax24.imshow(img23, cmap='gray')
    ax24.set_title('labels 5', fontsize=10)
    ax29.imshow(img24, cmap='gray')
    ax29.set_title('labels 6', fontsize=10)
    ax5.imshow(img25, cmap='gray')
    ax5.set_title('output 1', fontsize=10)
    ax10.imshow(img26, cmap='gray')
    ax10.set_title('output 2', fontsize=10)
    ax15.imshow(img27, cmap='gray')
    ax15.set_title('output 3', fontsize=10)
    ax20.imshow(img28, cmap='gray')
    ax20.set_title('output 4', fontsize=10)
    ax25.imshow(img29, cmap='gray')
    ax25.set_title('output 5', fontsize=10)
    ax30.imshow(img30, cmap='gray')
    ax30.set_title('output 6', fontsize=10)
    plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)
    plt.show()
   

def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation

def analyze_RGB(img):
    img = cv2.imread(fn)
    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    
    # Convert subsampled image to desired color space(s)
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_small_rgb = img_small / 255.  # scaled to [0, 1], only for plotting
    
    plot3d(img_small_RGB, img_small_rgb, axis_labels=list("RGB"))
    plt.show()
    
    img = mpimg.imread(fn)
    img_RGB = img
    r_chan = img_RGB[:,:,0]
    g_chan = img_RGB[:,:,1]
    b_chan = img_RGB[:,:,2]
    
    r = np.dstack((r_chan, r_chan, r_chan))
    g = np.dstack((g_chan, g_chan, g_chan))
    b = np.dstack((b_chan, b_chan, b_chan))
    
    inspect3(img, r, g, b)
    
def analyze_HSV(img):
    img = cv2.imread(fn)
    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    
    # Convert subsampled image to desired color space(s)
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    img_small_rgb = img_small / 255.  # scaled to [0, 1], only for plotting
    
    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    plt.show()
    
    img = mpimg.imread(fn)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_chan = img_HSV[:,:,0]
    s_chan = img_HSV[:,:,1]
    v_chan = img_HSV[:,:,2]
    
    h = np.dstack((h_chan, h_chan, h_chan))
    s = np.dstack((s_chan, s_chan, s_chan))
    v = np.dstack((v_chan, v_chan, v_chan))
    
    inspect3(img, h, s, v)

def analyze_HLS(img):
    img = cv2.imread(fn)
    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    
    # Convert subsampled image to desired color space(s)
    img_small_HLS = cv2.cvtColor(img_small, cv2.COLOR_BGR2HLS)
    img_small_rgb = img_small / 255.  # scaled to [0, 1], only for plotting
    
    plot3d(img_small_HLS, img_small_rgb, axis_labels=list("HLS"))
    plt.show()
    
    img = mpimg.imread(fn)
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_chan = img_HLS[:,:,0]
    l_chan = img_HLS[:,:,1]
    s_chan = img_HLS[:,:,2]
    
    h = np.dstack((h_chan, h_chan, h_chan))
    l = np.dstack((l_chan, l_chan, l_chan))
    s = np.dstack((s_chan, s_chan, s_chan))
    
    inspect3(img, h, l, s)

def analyze_LUV(img):
    img = cv2.imread(fn)
    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    
    # Convert subsampled image to desired color space(s)
    img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2LUV)
    img_small_rgb = img_small / 255.  # scaled to [0, 1], only for plotting
    
    plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"))
    plt.show()
    
    img = mpimg.imread(fn)
    img_LUV = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_chan = img_LUV[:,:,0]
    u_chan = img_LUV[:,:,1]
    v_chan = img_LUV[:,:,2]
    
    l = np.dstack((l_chan, l_chan, l_chan))
    u = np.dstack((u_chan, u_chan, u_chan))
    v = np.dstack((v_chan, v_chan, v_chan))
    
    inspect3(img, l, u, v)
    
def analyze_YUV(fn):
    img = cv2.imread(fn)
    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    
    # Convert subsampled image to desired color space(s)
    img_small_YUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2YUV)
    img_small_rgb = img_small / 255.  # scaled to [0, 1], only for plotting
    
    plot3d(img_small_YUV, img_small_rgb, axis_labels=list("YUV"))
    plt.show()
    
    img = mpimg.imread(fn)
    img_YUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y_chan = img_YUV[:,:,0]
    u_chan = img_YUV[:,:,1]
    v_chan = img_YUV[:,:,2]
    
    y = np.dstack((y_chan, y_chan, y_chan))
    u = np.dstack((u_chan, u_chan, u_chan))
    v = np.dstack((v_chan, v_chan, v_chan))
    
    inspect3(img, y, u, v)

def main():    

#     fn1 = "./output_images/project_video_output_14.png"
#     fn2 = "./output_images/project_video_output_31.png"
#     fn3 = "./output_images/project_video_output_32.png"
#     fn4 = "./output_images/project_video_output_38.png"
#     
#     img1 = mpimg.imread(fn1)
#     img2 = mpimg.imread(fn2)
#     img3 = mpimg.imread(fn3)
#     img4 = mpimg.imread(fn4)
#     inspect2x2(img1, img2, img3, img4)


    fn1 = "./output_images/project_video_windows_0.png"
    fn2 = "./output_images/project_video_windows_1.png"
    fn3 = "./output_images/project_video_windows_2.png"
    fn4 = "./output_images/project_video_windows_3.png"
    fn5 = "./output_images/project_video_windows_4.png"
    fn6 = "./output_images/project_video_windows_5.png"
    fn7 = "./output_images/project_video_heatmap0.png"
    fn8 = "./output_images/project_video_heatmap1.png"
    fn9 = "./output_images/project_video_heatmap2.png"    
    fn10 = "./output_images/project_video_heatmap3.png"
    fn11 = "./output_images/project_video_heatmap4.png"
    fn12 = "./output_images/project_video_heatmap5.png"
    fn13 = "./output_images/project_video_heatmap_thresh0.png"
    fn14 = "./output_images/project_video_heatmap_thresh1.png"
    fn15 = "./output_images/project_video_heatmap_thresh2.png"
    fn16 = "./output_images/project_video_heatmap_thresh3.png"
    fn17 = "./output_images/project_video_heatmap_thresh4.png"
    fn18 = "./output_images/project_video_heatmap_thresh5.png"
    fn19 = "./output_images/labels_1.png"
    fn20 = "./output_images/labels_2.png"
    fn21 = "./output_images/labels_3.png"
    fn22 = "./output_images/labels_4.png"
    fn23 = "./output_images/labels_5.png"
    fn24 = "./output_images/labels_6.png"
    fn25 = "./output_images/project_video_output_1.png"
    fn26 = "./output_images/project_video_output_2.png"
    fn27 = "./output_images/project_video_output_3.png"
    fn28 = "./output_images/project_video_output_4.png"
    fn29 = "./output_images/project_video_output_5.png"
    fn30 = "./output_images/project_video_output_6.png"

    img1 = mpimg.imread(fn1)
    img2 = mpimg.imread(fn2)
    img3 = mpimg.imread(fn3)
    img4 = mpimg.imread(fn4)
    img5 = mpimg.imread(fn5)
    img6 = mpimg.imread(fn6)
    img7 = mpimg.imread(fn7)
    img8 = mpimg.imread(fn8)
    img9 = mpimg.imread(fn9)
    img10 = mpimg.imread(fn10)
    img11 = mpimg.imread(fn11)
    img12 = mpimg.imread(fn12)
    img13 = mpimg.imread(fn13)
    img14 = mpimg.imread(fn14)
    img15 = mpimg.imread(fn15)
    img16 = mpimg.imread(fn16)
    img17 = mpimg.imread(fn17)
    img18 = mpimg.imread(fn18)
    img19 = mpimg.imread(fn19)
    img20 = mpimg.imread(fn20)
    img21 = mpimg.imread(fn21)
    img22 = mpimg.imread(fn22)
    img23 = mpimg.imread(fn23)
    img24 = mpimg.imread(fn24)
    img25 = mpimg.imread(fn25)
    img26 = mpimg.imread(fn26)
    img27 = mpimg.imread(fn27)
    img28 = mpimg.imread(fn28)
    img29 = mpimg.imread(fn29)
    img30 = mpimg.imread(fn30)
    inspect5x6(img1, img2, img3, img4, img5, img6, img7, img8, img9, img10,
               img11, img12, img13, img14, img15, img16, img17, img18, img19, img20,
               img21, img22, img23, img24, img25, img26, img27, img28, img29, img30)

#    analyze_YUV(fn14)
    
if __name__ == "__main__":
    main()