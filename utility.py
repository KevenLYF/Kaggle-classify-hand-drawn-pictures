import numpy as np
import cv2

# img: flatten array (no need to reshape)
# return: image (100 * 100) with noise removed, value is either 0 or 255
def cleanNoise(img):
    img = img.reshape(100, 100)
    for i in range(100):
        for j in range(100):
            if (img[i][j] > 200):
                img[i][j] = 255
            else:
                img[i][j] = 0

    img=img.astype(np.uint8)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    min_size = int(img.sum()/255*0.08)  # 0.08 is the relative ratio
    img2 = np.zeros((output.shape))

    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

def TrimImage(img):
    img_dim = img.shape
    up = 0
    down = img_dim[0]
    left = 0
    right = img_dim[1]

    for i in range(img_dim[0]):
        if (np.any(img[i, :] == 255)):
            break
        up += 1

    for i in range(img_dim[0]-1, 0, -1):
        if (np.any(img[i, :] == 255)):
            break
        down -= 1

    for i in range(img_dim[1]):
        if (np.any(img[:, i] == 255)):
            break
        left += 1

    for i in range(img_dim[1]-1, 0, -1):
        if (np.any(img[:, i] == 255)):
            break
        right -= 1

    if up > down or left > right:
        up = 0
        down = img_dim[0]
        left = 0
        right = img_dim[1]


    if down - up > right - left:
        diff = (down - up) - (right - left)
        right += int(diff/2)
        left -= int(diff/2)
    elif right - left > down - up:
        diff = (right - left) - (down - up)
        down += int(diff/2)
        up -= int(diff/2)
    

    if up < 0 :
        up = 0

    if left < 0 :
        left = 0

    if down > img_dim[0] :
        down = img_dim[0]

    if right > img_dim[1] :
        right = img_dim[1]
        
    result = img[up:down, left:right]
    result = cv2.resize(result, (100, 100)) 

    return result

def moveToMid(img):
    img_dim = img.shape
    up = 0
    down = img_dim[0]
    left = 0
    right = img_dim[1]

    for i in range(img_dim[0]):
        if (np.any(img[i, :] == 255)):
            break
        up += 1

    for i in range(img_dim[0] - 1, 0, -1):
        if (np.any(img[i, :] == 255)):
            break
        down -= 1

    for i in range(img_dim[1]):
        if (np.any(img[:, i] == 255)):
            break
        left += 1

    for i in range(img_dim[1] - 1, 0, -1):
        if (np.any(img[:, i] == 255)):
            break
        right -= 1

    if up > down or left > right:
        return img

    obj = img[up:down, left:right]
    width = right - left
    height = down - up
    verInterval = int((100-height)/2)
    horiInterval = int((100-width)/2)

    newImg = np.array([[0]*100]*100)
    for i in range(verInterval, verInterval + height):
        newImg[i, horiInterval: horiInterval+width] = obj[i-verInterval]

    return newImg