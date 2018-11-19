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

    result = img[up:down, left:right]
    result = cv2.resize(result, (100, 100)) 

    return result