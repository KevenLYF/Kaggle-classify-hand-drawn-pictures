# img:
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