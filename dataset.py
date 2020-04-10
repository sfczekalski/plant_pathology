from globals import *
from imports import *


class PlantDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_path = data_dir + 'images/' + self.df.loc[idx, 'image_id'] + '.jpg'
        image = load_image(image_path=image_path)

        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.unsqueeze(-1)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

            return image, labels
