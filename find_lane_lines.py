import numpy as np
import cv2
import glob
import pickle
import os.path
from  Line import *
from moviepy.editor import VideoFileClip


def calibrate_camera() :
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('camera_cal/*')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imwrite("output_images/find_corners_output.jpg", img);


    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    calibration_data = {}
    calibration_data["mtx"] = mtx
    calibration_data["dist"] = dist
    pickle.dump( calibration_data, open( "calibration_data.p", "wb" ) )


    for fname in images:
        print (fname)
        img = cv2.imread(fname)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('output_images/'+fname, dst)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Define a function to take x,y vectors of arbitrary length and warp to new space given transform matrix M
def warp_points(x, y, M):
    # Expect x and y to be vectors and transform matrix M to be a 3x3
    xnew = (M[0][0]*x + M[0][1]*y + M[0][2])/(M[2][0]*x + M[2][1]*y + M[2][2])
    ynew = (M[1][0]*x + M[1][1]*y + M[1][2])/(M[2][0]*x + M[2][1]*y + M[2][2])

    return xnew, ynew


def find_line_pixels(binimg):

    # Define some parameters for line search, if lines not already detected
    img_shape = binimg.shape
    search_smooth = 11
    ystart_frac = 0.5
    ymid = np.int(img_shape[0]/2)
    xmid = np.int(img_shape[1]/2)

    margin = 50 # width to include pixels left and right of the fit
    min_detected_pixels = 10 #arbitrary cutoff for minimum number of pixels defining a line

    # Identify all nonzero pixels in the binary image
    nonzero = binimg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # If the left line was detected in the previous frame, search for new detection within +/- margin in x
    if left_line.detected:
        leftfit = left_line.best_fit
        # Identify all pixels associated with the lines
        good_inds = ((nonzerox > (leftfit[0]*(nonzeroy**2) + leftfit[1]*nonzeroy + leftfit[2] - margin)) & \
                    (nonzerox < (leftfit[0]*(nonzeroy**2) + leftfit[1]*nonzeroy + leftfit[2] + margin)))

        if len(good_inds) > min_detected_pixels:
            left_line.allx = nonzerox[good_inds]
            left_line.ally = nonzeroy[good_inds]
            left_fit = np.polyfit(left_line.ally, left_line.allx, 2)
            left_line.detected = True
        else:
            left_line.detected = False

    else:
        # Search the lower left quadrant of the image for the left line base
        counts = np.sum(binimg[ymid:, 0:xmid], axis=0) #histogram of hot pixels in each column of pixels
        smooth_counts = np.convolve(counts, np.ones(search_smooth)/search_smooth, mode='same') #boxcar smooth
        line_cen = np.argmax(smooth_counts)
        pleftx = [line_cen - margin, line_cen + margin]
        good_inds = line_search(nonzerox, nonzeroy, pleftx, ystart_frac, ymid, img_shape, margin)

        if len(good_inds) > min_detected_pixels:
            left_line.allx = nonzerox[good_inds]
            left_line.ally = nonzeroy[good_inds]
            left_fit = np.polyfit(left_line.ally, left_line.allx, 2)
            left_line.detected = True
        else:
            left_line.detected = False

    # If the right line was detected in the previous frame, search for new detection within +/- margin in x
    if right_line.detected:
        rightfit = right_line.best_fit
        # Identify all pixels associated with the lines
        good_inds = ((nonzerox > (rightfit[0]*(nonzeroy**2) + rightfit[1]*nonzeroy + rightfit[2] - margin)) & \
                     (nonzerox < (rightfit[0]*(nonzeroy**2) + rightfit[1]*nonzeroy + rightfit[2] + margin)))
        if len(good_inds) > min_detected_pixels:
            right_line.allx = nonzerox[good_inds]
            right_line.ally = nonzeroy[good_inds]
            right_fit = np.polyfit(right_line.ally, right_line.allx, 2)
            right_line.detected = True
        else:
            right_line.detected = False

    else:
        # Search the lower left quadrant of the image for the left line base
        print('initializing search for the right line...')
        counts = np.sum(binimg[ymid:, xmid:], axis=0) #histogram of hot pixels in each column of pixels
        smooth_counts = np.convolve(counts, np.ones(search_smooth)/search_smooth, mode='same')
        line_cen = np.argmax(smooth_counts) + xmid
        prightx = [line_cen - margin, line_cen + margin]
        good_inds = line_search(nonzerox, nonzeroy, prightx, ystart_frac, ymid, img_shape, margin)

        if len(good_inds) > min_detected_pixels:
            right_line.allx = nonzerox[good_inds]
            right_line.ally = nonzeroy[good_inds]
            right_fit = np.polyfit(right_line.ally, right_line.allx, 2)
            right_line.detected = True
        else:
            right_line.detected = False


    recent_detections = 10 #an arbitrary number of recent detections to take into consideration when deciding
                            #whether a new detection is an outlier
    if left_line.detected:

        # Take the difference in fit coefficients between last fit and new fit
        diff_left = np.absolute(left_line.current_fit - left_fit)
        left_line.diffs = np.vstack((left_line.diffs, diff_left))

        # If we have enough data do the outlier rejection step
        if left_line.diffs.shape[0] > recent_detections:
            left_std = np.std(left_line.diffs[:,0])
            left_line.diffs = left_line.diffs[-recent_detections:]

            # Reject outliers where quadratic term of the fit is > 4 standard devs from best fit
            if diff_left[0] < 4*left_std:
                left_line.current_fit = left_fit
        else:
            left_line.current_fit = left_fit

    if right_line.detected:

        # Take the difference in fit coefficients between last fit and new fit
        diff_right = np.absolute(right_line.current_fit - right_fit)
        right_line.diffs = np.vstack((right_line.diffs, diff_right))

        if right_line.diffs.shape[0] > recent_detections:
            right_std = np.std(right_line.diffs[:,0])
            right_line.diffs = right_line.diffs[-recent_detections:]
            if diff_right[0] < 4*right_std:
                right_line.current_fit = right_fit
        else:
            right_line.current_fit = right_fit


    y_val = img_shape[0]*0.5
    left_curverad = ((1 + (2*left_fit[0]*y_val + left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_val + right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])
    pixels_to_meters = 720/30  #this is wrong
    avg_curverad = (left_curverad + right_curverad)/2/pixels_to_meters
    left_line.radius_of_curvature = avg_curverad
    left_line.deg_per_meter = 360/(2*np.pi*avg_curverad)

def find_newgood(nzerox, nzeroy, startind, stopind, width, fit):

    # Extrapolate a fit to the lane line to include new pixels
    fitdeg = len(fit) - 1
    if fitdeg == 2:
        xfit = fit[0]*nzeroy**2 + fit[1]*nzeroy + fit[2]
    elif fitdeg == 1:
        xfit = fit[0]*nzeroy + fit[1]
    good = ((nzerox >= (xfit - width)) \
                    & (nzerox <= (xfit + width)) \
                    & (nzeroy > startind) \
                    & (nzeroy <= stopind)).nonzero()[0]
    return good

def line_search(nzx, nzy, px, ystart_frac, startind, img_shape, margin):

    # Identify pixels associated with the base of the line (bottom of the image)
    good1 = ((nzx >= px[0]) & (nzx <= px[1]) & (nzy >= startind)).nonzero()[0]
    nonzero1x = nzx[good1]
    nonzero1y = nzy[good1]

    # Define the starting point for bins in y to iteratively find the lines
    next_ysfrac = ystart_frac - 0.1
    last_ysfrac = 0
    nsteps = 1 + next_ysfrac/0.1
    ystarts = np.linspace(next_ysfrac, last_ysfrac, nsteps, endpoint=True)
    dy = 1 - ystart_frac # y bin size

    for ys in ystarts:  # step through the bins in y

        order = 1    # order of the polynomial fit
        line1 = np.polyfit(nonzero1y, nonzero1x, order) # fit a line to pixels in the base of left line
        startind = round(img_shape[0]*ys)   # start point of y bin
        stopind = round(img_shape[0]*(ys+dy))  # end point of y bin
        #select new set of pixels to add to the line
        new_good1 = find_newgood(nzx, nzy, startind, stopind, margin, line1)
        if len(new_good1) > 0:  #if we found some, add them to the line
            if len(new_good1) > 10: #if we found more than 10 use them to define next line segment selection
                nonzero1x = nzx[new_good1]
                nonzero1y = nzy[new_good1]

            good1 = np.hstack((good1, new_good1))


    # Get rid of duplicates from overlapping bins
    return np.unique(good1)

def warper(binary_img):

    # Define source and destination  coordinates
    img_size = (binary_img.shape[1], binary_img.shape[0])
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

    # Compute and apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv =  cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(binary_img, M, img_size, flags=cv2.INTER_NEAREST)

    return Minv, warped

# Define the main function for finding the lines, either based on previous detection or blind search
def find_line_pixels(binimg):

    # Define some parameters for line search, if lines not already detected
    img_shape = binimg.shape
    search_smooth = 11
    ystart_frac = 0.5
    ymid = np.int(img_shape[0]/2)
    xmid = np.int(img_shape[1]/2)

    margin = 50 # width to include pixels left and right of the fit
    min_detected_pixels = 10 #arbitrary cutoff for minimum number of pixels defining a line

    # Identify all nonzero pixels in the binary image
    nonzero = binimg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # If the left line was detected in the previous frame, search for new detection within +/- margin in x
    if left_line.detected:
        leftfit = left_line.best_fit
        # Identify all pixels associated with the lines
        good_inds = ((nonzerox > (leftfit[0]*(nonzeroy**2) + leftfit[1]*nonzeroy + leftfit[2] - margin)) & \
                    (nonzerox < (leftfit[0]*(nonzeroy**2) + leftfit[1]*nonzeroy + leftfit[2] + margin)))

        if len(good_inds) > min_detected_pixels:
            left_line.allx = nonzerox[good_inds]
            left_line.ally = nonzeroy[good_inds]
            left_fit = np.polyfit(left_line.ally, left_line.allx, 2)
            left_line.detected = True
        else:
            left_line.detected = False

    else:
        # Search the lower left quadrant of the image for the left line base
        print('initializing search for left line...')
        counts = np.sum(binimg[ymid:, 0:xmid], axis=0) #histogram of hot pixels in each column of pixels
        smooth_counts = np.convolve(counts, np.ones(search_smooth)/search_smooth, mode='same') #boxcar smooth
        line_cen = np.argmax(smooth_counts)
        pleftx = [line_cen - margin, line_cen + margin]
        good_inds = line_search(nonzerox, nonzeroy, pleftx, ystart_frac, ymid, img_shape, margin)

        if len(good_inds) > min_detected_pixels:
            left_line.allx = nonzerox[good_inds]
            left_line.ally = nonzeroy[good_inds]
            left_fit = np.polyfit(left_line.ally, left_line.allx, 2)
            left_line.detected = True
        else:
            left_line.detected = False

    # If the right line was detected in the previous frame, search for new detection within +/- margin in x
    if right_line.detected:
        rightfit = right_line.best_fit
        # Identify all pixels associated with the lines
        good_inds = ((nonzerox > (rightfit[0]*(nonzeroy**2) + rightfit[1]*nonzeroy + rightfit[2] - margin)) & \
                     (nonzerox < (rightfit[0]*(nonzeroy**2) + rightfit[1]*nonzeroy + rightfit[2] + margin)))
        if len(good_inds) > min_detected_pixels:
            right_line.allx = nonzerox[good_inds]
            right_line.ally = nonzeroy[good_inds]
            right_fit = np.polyfit(right_line.ally, right_line.allx, 2)
            right_line.detected = True
        else:
            right_line.detected = False

    else:
        # Search the lower left quadrant of the image for the left line base
        print('initializing search for the right line...')
        counts = np.sum(binimg[ymid:, xmid:], axis=0) #histogram of hot pixels in each column of pixels
        smooth_counts = np.convolve(counts, np.ones(search_smooth)/search_smooth, mode='same') #boxcar smooth
        line_cen = np.argmax(smooth_counts) + xmid
        prightx = [line_cen - margin, line_cen + margin]
        good_inds = line_search(nonzerox, nonzeroy, prightx, ystart_frac, ymid, img_shape, margin)

        if len(good_inds) > min_detected_pixels:
            right_line.allx = nonzerox[good_inds]
            right_line.ally = nonzeroy[good_inds]
            right_fit = np.polyfit(right_line.ally, right_line.allx, 2)
            right_line.detected = True
        else:
            right_line.detected = False


    recent_detections = 10 #an arbitrary number of recent detections to take into consideration when deciding
                            #whether a new detection is an outlier
    if left_line.detected:

        # Take the difference in fit coefficients between last fit and new fit
        diff_left = np.absolute(left_line.current_fit - left_fit)
        left_line.diffs = np.vstack((left_line.diffs, diff_left))

        # If we have enough data do the outlier rejection step
        if left_line.diffs.shape[0] > recent_detections:
            left_std = np.std(left_line.diffs[:,0])
            left_line.diffs = left_line.diffs[-recent_detections:]

            # Reject outliers where quadratic term of the fit is > 4 standard devs from best fit
            if diff_left[0] < 4*left_std:
                left_line.current_fit = left_fit
        else:
            left_line.current_fit = left_fit

    if right_line.detected:

        # Take the difference in fit coefficients between last fit and new fit
        diff_right = np.absolute(right_line.current_fit - right_fit)
        right_line.diffs = np.vstack((right_line.diffs, diff_right))

        if right_line.diffs.shape[0] > recent_detections:
            right_std = np.std(right_line.diffs[:,0])
            right_line.diffs = right_line.diffs[-recent_detections:]
            if diff_right[0] < 4*right_std:
                right_line.current_fit = right_fit
        else:
            right_line.current_fit = right_fit


    y_vals = img_shape[0]*0.5
    left_curverad = ((1 + (2*left_fit[0]* y_vals + left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]* y_vals + right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])
    pixels_to_meters = 720/30
    avg_curverad = (left_curverad + right_curverad)/2/pixels_to_meters
    left_line.radius_of_curvature = avg_curverad
    left_line.deg_per_meter = 360/(2*np.pi*avg_curverad)


def pipeline(image, image_name = ''):

    if not os.path.exists("calibration_data.p") :
        calibrate_camera()

    calibration_data = pickle.load( open( "calibration_data.p", "rb" ) )
    mtx = calibration_data["mtx"]
    dist = calibration_data["dist"]

    imshape = image.shape
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    if image_name :
        cv2.imwrite('output_images/distortion_correction/'+image_name, dst);

    vertices = np.array([[(150,imshape[0]),(imshape[1]/2, imshape[0]/2 + 30),
                          (imshape[1]/2, imshape[0]/2 + 30), (imshape[1]-50,imshape[0])]], dtype=np.int32)
    masked_dst = region_of_interest(dst, vertices) # and masking


    hls = cv2.cvtColor(masked_dst, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize =3) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x-gradient and s channel for lane pixel selection
    retval, sxbinary = cv2.threshold(scaled_sobel.astype('uint8'), 10, 100, cv2.THRESH_BINARY)

    retval, s_binary = cv2.threshold(s_channel.astype('uint8'), 190, 230, cv2.THRESH_BINARY)

    combined_image = cv2.bitwise_or(s_binary, sxbinary)

    if image_name :
        cv2.imwrite('output_images/thresholded_binary/'+image_name, combined_image)



    # Perspective transform
    Minv, warped = warper(combined_image)

    if image_name :
        cv2.imwrite('output_images/perspective_transform/'+image_name, warped)

    # Find lane pixels
    find_line_pixels(warped)

    # calculate radius of curvature
    yfit = np.linspace(0, imshape[0] - 1, imshape[0]).astype('int')
    lfit = left_line.current_fit
    rfit = right_line.current_fit
    left_xfit = np.clip(np.round(lfit[0]*yfit**2 + lfit[1]*yfit + lfit[2]).astype('int'), 0, imshape[1]-1)
    right_xfit = np.clip(np.round(rfit[0]*yfit**2 + rfit[1]*yfit + rfit[2]).astype('int'), 0, imshape[1]-1)

    # Update the best fit if new lines were detected

    smooth = 5
    if left_line.detected:
        if len(left_line.recent_xfitted) > 0:
            left_line.recent_xfitted.append(left_xfit)
        else:
            left_line.recent_xfitted = [left_xfit]

        if len(left_line.recent_xfitted) > smooth:
            left_line.recent_xfitted = left_line.recent_xfitted[-smooth:]
        left_line.bestx = np.mean(np.array(left_line.recent_xfitted), axis=0).astype('int')
        left_line.best_fit = np.polyfit(yfit, left_line.bestx, 2)
        left_line.line_xvals = left_xfit #grab the left position of the most recent lane
        left_line.line_yvals = yfit

    if right_line.detected:
        if len(right_line.recent_xfitted) > 0:
            right_line.recent_xfitted.append(right_xfit)
        else:
            right_line.recent_xfitted = [right_xfit]

        if len(right_line.recent_xfitted) > smooth:
            right_line.recent_xfitted = right_line.recent_xfitted[-smooth:]
        right_line.bestx = np.mean(np.array(right_line.recent_xfitted), axis=0).astype('int')
        right_line.best_fit = np.polyfit(yfit, right_line.bestx, 2)
        right_line.line_xvals = right_xfit #grab the right position of the most recent lane
        right_line.line_yvals = yfit




    # Generating some "slides" to plot color lines
    warp_copy1 = np.zeros(warped.shape, dtype='uint8')
    warp_copy2 = np.zeros(warped.shape, dtype='uint8')
    warp_copy3 = np.zeros(warped.shape, dtype='uint8')

    # plot right and left lines

    warp_copy2[left_line.ally, left_line.allx] = 255
    warp_copy3[right_line.ally, right_line.allx] = 255
    color_binary = np.dstack((warp_copy2, warp_copy1, warp_copy3))
    color_warped = np.dstack((warped, warped, warped))
    binresult = cv2.addWeighted(color_warped, 0.7, color_binary, 1, 0)
    if image_name :
        cv2.imwrite('output_images/lane_lines_pixels/'+image_name, binresult)

    # Unwarp slides back to image space
    unwarp_copy1 = cv2.warpPerspective(warp_copy1, Minv,
                                       (imshape[1], imshape[0]),
                                       flags=cv2.INTER_NEAREST)
    unwarp_copy2 = cv2.warpPerspective(warp_copy2, Minv,
                                       (imshape[1], imshape[0]),
                                       flags=cv2.INTER_NEAREST)
    unwarp_copy3 = cv2.warpPerspective(warp_copy3, Minv,
                                       (imshape[1], imshape[0]),
                                       flags=cv2.INTER_NEAREST)

    # Unwarp fitted line pixels
    unwarp_xpix_left, unwarp_ypix  = warp_points(left_line.bestx, yfit, Minv)
    unwarp_xpix_right, unwarp_ypix  = warp_points(right_line.bestx, yfit, Minv)
    left_line.line_base_pos = unwarp_xpix_left[-1]
    right_line.line_base_pos = unwarp_xpix_right[-1]

    # Find top and bottom of the fitted lines
    top = np.min(unwarp_ypix)
    bottom = np.max(unwarp_ypix)

    # Create an array of y points to interpolate the fit in image space
    n_ypts = bottom - top + 1
    interp_y = np.linspace(top, bottom, n_ypts).astype('int')
    interp_xleft = np.round(np.interp(interp_y, unwarp_ypix, unwarp_xpix_left)).astype('int')
    interp_xright = np.round(np.interp(interp_y, unwarp_ypix, unwarp_xpix_right)).astype('int')

    # Get our fitted lane lines into shape for cv2.fillPoly
    pts_left = np.array([np.transpose(np.vstack([interp_xleft, interp_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([interp_xright, interp_y])))])
    pts = np.hstack((pts_left, pts_right))


    # Create some slides to draw lane on
    line_drawn1 = np.zeros_like(unwarp_copy1)
    line_drawn2 = np.zeros_like(unwarp_copy1)
    line_drawn3 = np.zeros_like(unwarp_copy1)

    # Create a mask to fill the area between the left and right lines
    cv2.fillPoly(line_drawn2, pts, 255)

    if image_name :
        cv2.imwrite('output_images/lane_lines_detected/'+image_name, line_drawn2)

    # Create some color images to show lane detection
    # First, just a green polygon covering the lane area
    lane_unwarp = np.dstack((line_drawn1, line_drawn2, line_drawn3)).astype('uint8')
    # Show the left lane pixels in blue and the right lane pixels in red
    leftright_unwarp = np.dstack((unwarp_copy2, line_drawn1, unwarp_copy3)).astype('uint8')
    # Paint the lines and lane area back on the road
    line_result = cv2.addWeighted(dst, 1, lane_unwarp, 0.3, 0)
    line_lr_result = cv2.addWeighted(line_result, 0.7, leftright_unwarp, 1, 0)

    # Text output for radius of curvature and vehicle position
    curve_string = str(np.int(left_line.radius_of_curvature))
    cv2.putText(line_lr_result,"Radius of Curvature = " + curve_string + '(m)', (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    center = image.shape[1]/2
    pix_to_meter = 1280/3
    vehicle_pos = round((center - (left_line.line_base_pos+right_line.line_base_pos)/2)/pix_to_meter, 2)
    if vehicle_pos < 0:
        pos_string = ' left of center'
    else:
        pos_string = ' right of center'

    cv2.putText(line_lr_result,"Vehicle is " + str(np.absolute(vehicle_pos)) + 'm' + pos_string, (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    return line_lr_result


if __name__ == '__main__':
    images = glob.glob('test_images/*')
    for fname in images:
        left_line = Line()
        right_line = Line()
        image = cv2.imread(fname)
        image_name = fname.split('/')[1]
        result = pipeline(image, image_name)
        cv2.imwrite('output_images/final_output/'+image_name, result);

    project_video_output = 'project_video_output.mp4'
    clip = VideoFileClip("project_video.mp4")

    project_video_clip = clip.fl_image(pipeline)
    project_video_clip.write_videofile(project_video_output, audio=False)
