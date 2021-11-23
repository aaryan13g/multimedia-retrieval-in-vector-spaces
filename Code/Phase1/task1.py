import numpy as np
from scipy.stats import skew
import sys
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern, hog
from skimage import exposure
from skimage.transform import resize


def get_sys_args():
    return str(sys.argv[1]), str(sys.argv[2])


def fetch_image(imageID):
    image_path = imageID
    image = Image.open(image_path)
    image_data = np.asarray(image)
    image_data = image_data / 255   # normalize the image array
    return image_data


def split_into_windows(image_data, window_size=(8, 8)):
    ht, wd = image_data.shape
    data = image_data.copy()
    return data.reshape(ht // window_size[0], window_size[0], -1, window_size[1]).swapaxes(1, 2).reshape(-1, window_size[0], window_size[1])


def cal_skew(window, mean):
    total = 0
    for p in range(len(window)):
        for q in range(len(window[p])):
            total = total + pow((window[p][q] - mean), 3)
    total = total / 64

    total = pow(abs(total), float(1) / 3) * (1, -1)[total < 0]
    return total


def color_moments_model(image_data):
    windows = split_into_windows(image_data)
    cm_descriptor = []
    for window in windows:
        mean = np.mean(window)
        std = np.std(window)
        # skewness = skew(window, axis=None)
        skewness = cal_skew(window, mean)
        color_moments = [mean, std, skewness]
        cm_descriptor.append(color_moments)

    cm_descriptor = np.array(cm_descriptor)
    return cm_descriptor


def visualize_color_moments_model(cm_descriptor):
    x_pos = np.arange(cm_descriptor.shape[0])
    means = list(cm_descriptor[:, 0])
    stds = list(cm_descriptor[:, 1])
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=stds, alpha=0.8, ecolor='black', capsize=2)
    ax.set_xticks(x_pos)
    ax.set_xlabel("Windows")
    ax.set_title('Mean and Standard Deviation')
    ax.yaxis.grid(True)
    ax.plot(x_pos, list(cm_descriptor[:, 2]))
    plt.tight_layout()
    plt.show()


def extended_local_binary_patterns_model(image_data):
    lbp_ror = local_binary_pattern(image_data/255, 8, 1, method="uniform")
    # lbp_var = local_binary_pattern(image_data, 8, 1, method="var")
    # elbp = lbp_ror * lbp_var
    return lbp_ror




def extract_lbp(image):
        """
        Method to extract LBP
        """
        #Parameters for LBP
        NUM_POINTS = 8
        RADIUS = 1
        METHOD_UNIFORM = 'uniform'
        WINDOW_SIZE = 8
        BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        
        lbp_list = np.asarray([])

        # splits the 64*64 image into 64 blocks of 8*8 pixels and calculates lbp vector for each block
        for row in range(0, image.shape[0], WINDOW_SIZE):
            for column in range(0, image.shape[1], WINDOW_SIZE):
                window = image[row:row + WINDOW_SIZE, column:column + WINDOW_SIZE]
                lbp = local_binary_pattern(
                    window, NUM_POINTS, RADIUS, METHOD_UNIFORM)  
                window_histogram = np.histogram(lbp, bins=BINS)[0]
                lbp_list = np.append(lbp_list, window_histogram)
        lbp_list.ravel()
        return lbp_list.tolist()



def visualize_elbp_model(lbp_descriptor):
    im = Image.fromarray(np.uint8(lbp_descriptor), "L")
    im.show()
    im.save("../Outputs/task1_ELBP.png")


def histogram_of_oriented_gradients_model(image_data):
    image_data = resize(image_data, (128, 64))
    hog_desc, hog_image = hog(image_data, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True)
    return hog_desc, hog_image


def visualize_hog_model(hog_desc, hog_image):
    hog_image_rescaled = exposure.rescale_intensity(np.uint8(hog_image * 255), in_range=(0, 10))
    im = Image.fromarray(hog_image_rescaled, "L")
    im.show()
    im.save("../Outputs/task1_HOG_img.png")
    plt.hist(hog_desc, bins=255)
    plt.show()


if __name__ == "__main__":
    imageID, model = get_sys_args()
    image_data = fetch_image(imageID)
    if model in ("CM", "cm", "cm8x8", "CM8x8"):
        descriptor = color_moments_model(image_data)
        print(descriptor)
        visualize_color_moments_model(descriptor)
    elif model == "ELBP" or model == "elbp":
        #descriptor = extended_local_binary_patterns_model(image_data)
        descriptor = extract_lbp(image_data)
        print(descriptor)
        #visualize_elbp_model(descriptor)
    elif model == "HOG" or model == "hog":
        descriptor, hog_image = histogram_of_oriented_gradients_model(image_data)
        print(descriptor)
        visualize_hog_model(descriptor, hog_image)
    else:
        print("Please enter correct model name.")

