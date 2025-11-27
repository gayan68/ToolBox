import cv2  # OpenCV (often python-opencv on Arch)
import numpy as np
import matplotlib.pyplot as plt

def find_edge(image, user_threshold=None, show_density=False, action="Crop"):
    """
    Detect the document region in the image and either crop it or clean the borders.
    This function searches for borders starting from the center of the image and moving
    toward the top, bottom, left, and right until it finds a border (a significant color change).

    Args:
        image: The input image in OpenCV format.
        user_threshold: By default, the threshold is calculated automatically.
                        If a value is specified by the user, it will be used instead.
        show_density: If True, displays the image density graphs both column-wise and row-wise.
        action: If set to "Crop", the function removes the borders.
                Otherwise, it colors the borders with the minimum pixel value in the image
                (usually black).
    
    Returns:
        The processed image.

    Author: Gayan Pathirage
    """
    for axis in [0,1]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        axis_inv = 1 if axis == 0 else 0
        look_back=30
        
        side_sum = np.sum(gray, axis=axis_inv) 
        side_sum = side_sum / gray.shape[axis_inv]
        crop_min = int(gray.shape[axis]*.2)
        
        ### Auto threshold ###
        if user_threshold is None:
            thresh_max_val = abs(np.mean(side_sum)-np.max(side_sum))
            thresh_min_val = abs(np.mean(side_sum)-np.min(side_sum))
            threshold = np.max([thresh_max_val, thresh_min_val])//2
        else:
            threshold=user_threshold
        # print(threshold)
    
        find_right = True
        right = gray.shape[axis]
        old_val = side_sum[gray.shape[axis]//2+crop_min]
        for f in range(gray.shape[axis]//2+crop_min , gray.shape[axis]): 
            # if abs(side_sum[f]-old_val)>20:
            #     print(abs(side_sum[f]-old_val))
            if find_right and abs(side_sum[f]-old_val)>threshold:
                right = f
                find_right = False
            old_val = np.mean(side_sum[f-(look_back+10):f-look_back])
                
        find_left = True
        left = 0
        old_val = side_sum[gray.shape[axis]//2-crop_min]
        for b in range(gray.shape[axis]//2-crop_min , 0, -1):
            # if abs(side_sum[b]-old_val)>threshold:
            #     print(abs(side_sum[b]-old_val), " | " , side_sum[b], " ", old_val)
            if find_left and abs(side_sum[b]-old_val)>threshold:
                left = b
                find_left = False
            old_val = np.mean(side_sum[b+look_back:b+(look_back+10)])
    
        # print(f"left: {left}, right: {right}")

        if action == "Crop":
            ### Remove the borders in the image
            if axis == 0:
                cropped_image = image[left:right,:,:]
            else:
                cropped_image = image[:,left:right,:]
            image = cropped_image

        else:
            ### Paint the borders with lowest color in the document (Black)
            if axis == 0:
                image[0:left,:,:] = np.min(side_sum)
                image[right:gray.shape[axis],:,:] = np.min(side_sum)
            else:
                image[:,0:left,:] = np.min(side_sum)
                image[:,right:gray.shape[axis],:] = np.min(side_sum)
            cropped_image = image

        
        if show_density:
            plt.figure(figsize=(10,4))
            plt.plot(side_sum)
            plt.xlabel("Axis index")
            plt.ylabel("Sum of pixel values")
            plt.title(f"Axis {axis} Sum of Image")
            plt.show()
    return cropped_image
