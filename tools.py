import cv2


def load_image(image_path):
    """
    Load an image
    :param image_path: path (with image name) of the file
    :return: np.ndarray of shape=(CH, W, H) - the image
    """
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
