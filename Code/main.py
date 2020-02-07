import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np

# sum of squared differences
def compute_ssd(A,B):
  return  np.sum((A[:,:]-B[:,:])**2)


# sum of absolute differences
def compute_sad(A,B):
  return  np.sum(np.absolute(A[:,:]-B[:,:]))


# normalize np array to 0-225 range
def normlalize(a):
  b = np.copy(a)
  minimum = np.amin(a)
  maximum = np.amax(a)
  for i in range(0, a.shape[0]):
      for j in range(0, a.shape[1]):
          b[i,j] = (((1.0*(b[i,j] -minimum)/1.0)*(maximum-minimum))* 255)
  b = np.array(b, dtype= np.uint8)
  return b




'''
Find patch in strip and return column index (x value) of topleft corner
  Parameters:
    patch -> block in left strip
    strip -> right strip
  Returns:
    best_x -> column index (x value) of topleft corner from correct  right image patch
'''
def find_best_match(patch, strip, ssd=True):
  min_diff = 9999
  best_x = 0
  diff = 0
  
  for x in range(0,(strip.shape[1] - patch.shape[1])+1):
    other_patch = strip[:,x:(x + patch.shape[1])]
  
    if ssd == True:
      #sum of square differences
      diff = compute_ssd(patch,other_patch)
    else:
      #sum of absoulte differences
      diff = compute_sad(patch,other_patch)
    
    if diff < min_diff:
      min_diff = diff
      best_x = x

  return best_x



'''
Match two strips to compute disparity values 
  Parameters:
    strip_left -> left strip
    strip_right -> right strip
    b -> block/window size
  Returns:
    disparity -> vector of disparity for corresponding strip's row
'''
def match_strips(strip_left, strip_right, b):
  # num_blocks = len(strip_left[1]) - b 
  num_blocks = (strip_left.shape[1] - b)+1
  disparity = np.zeros(num_blocks, dtype=int)
  for block in range(0,num_blocks-1):
    x_left = block
   # patch_left = strip_left[:,x_left:(x_left +b -1)]
    patch_left = strip_left[:,x_left:(x_left +b )]
    # print(patch_left.shape)
    # print(patch_left.size)
    x_right = find_best_match(patch_left,strip_right)
    # disparity[block] = ((x_left - x_right)*30)
    disparity[block] = (x_left - x_right)
    # print("left: ",x_left," right: ",x_right)
  return disparity


'''
Compute disparity map between 2 images
Parameters:
  left_name  -> left image name
  right_name -> right image name
  w -> window size 
Returns:
  final_disparity -> full disparity map
'''
def disparity_map (left_name, right_name, w):
  
  ### Load images
  left =  cv2.imread(left_name)
  right = cv2.imread(right_name)
  
  ### Convert to grayscale,normalize values in range [0, 1] range for easier computation
  left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
  left_gray_norm = cv2.normalize(left_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
  right_gray_norm = cv2.normalize(right_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  # map to return
  final_disparity = np.empty((0,left_gray_norm.shape[1]-w+1), dtype=np.uint8)

  ### iterate on each scanline (row) [overlapping]
  b = w
  for y in range (0, (len(left_gray_norm[0]) - b)+1):
    
    ### Extract strip from left image
    strip_left = left_gray_norm[y:(y + b), :]
    # print("left strip shape: ",strip_left.shape)

    ### Extract strip from right image
    strip_right = right_gray_norm[y:(y + b ), :]
    # print("right strip shape: ",strip_right.shape)

    ### Now match these two strips to compute disparity values
    disparity_row = match_strips(strip_left, strip_right, b)
    # print("row")
    # print(disparity_row)
    # print("disparity row shape: ",disparity_row.shape) 
    # print("Disparity for image row ",," done !")

    #update disparity map with the new disparity row
    final_disparity = np.append(final_disparity, [disparity_row], axis=0 )

  #normalize to 0-255 values
  final_disparity = normlalize(final_disparity)

  return final_disparity



'''
Test Main
'''

window_size = [1, 5, 9]

##calculate disparity map
disparity = disparity_map('game_l.jpg','game_r.jpg',9)
# disparity = disparity_map('tsukuba_l.png','tsukuba_r.png',5)

#save disparity map to disk
np.save('disparity_map', disparity)

## plot disparity map 
print("Disparity Map")
#plt.imshow(disparity)
plt.imshow(disparity,'gray')
plt.show()