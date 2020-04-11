from globals import *
from dataset import PlantDataset


def load_image(image_path):
    """
    Load an image
    :param image_path: path (with image name) of the file
    :return: np.ndarray of shape=(CH, W, H) - the image
    """
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_batch():
    """
    Plot a batch of 9 from training set
    :return: None
    """
    batch_size = 9

    train_df = pd.read_csv(data_dir + 'train.csv')
    plotting_dataset = PlantDataset(train_df)
    plotting_dataloader = DataLoader(plotting_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    batch = next(iter(plotting_dataloader))
    images = batch[0]
    labels = batch[1]

    plt.figure(figsize=(16, 12))
    for i in range(batch_size):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title(labels[i])

    plt.suptitle('A batch from training set', fontsize=16)
    plt.show()


plot_batch()
