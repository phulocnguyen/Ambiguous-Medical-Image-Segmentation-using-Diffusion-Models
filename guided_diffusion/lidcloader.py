# import torch
# import torch.nn
# import numpy as np
# import os
# import os.path
# import nibabel
# import torch as th
# from skimage import io
# import random
# from matplotlib import pyplot as plt

# class LIDCDataset(torch.utils.data.Dataset):
#     def __init__(self, directory, test_flag=True):
#         '''
#         directory is expected to contain some folder structure:
#                   if some subfolder contains only files, all of these
#                   files are assumed to have a name like
#                   brats_train_001_XXX_123_w.nii.gz
#                   where XXX is one of t1, t1ce, t2, flair, seg
#                   we assume these five files belong to the same image
#                   seg is supposed to contain the segmentation
#         '''
#         super().__init__()
#         self.directory = os.path.expanduser(directory)

#         self.test_flag=test_flag
#         if test_flag:
#             self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']
#         else:
#             self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']

#         self.seqtypes_set = set(self.seqtypes)
#         self.database = []
#         for root, dirs, files in os.walk(self.directory):
#             # if there are no subdirs, we have data
#             if not dirs:
#                 files.sort()
#                 datapoint = dict()
#                 # extract all files as channels
#                 for f in files:
#                     seqtype = f.split('_')[0]
#                     #print(seqtype)
#                     datapoint[seqtype] = os.path.join(root, f)
#                     #print(datapoint)
#                 assert set(datapoint.keys()) == self.seqtypes_set, \
#                     f'datapoint is incomplete, keys are {datapoint.keys()}'
#                 self.database.append(datapoint)a
                

#     def __getitem__(self, x):
#         out = []
#         filedict = self.database[x]
#         for seqtype in self.seqtypes:
#             img = io.imread(filedict[seqtype])
#             img = img / 255
#             #nib_img = nibabel.load(filedict[seqtype])
#             path=filedict[seqtype]
#             out.append(torch.tensor(img))
#         out = torch.stack(out)
#         if self.test_flag:
#             image = out[0]
#             image = torch.unsqueeze(image, 0)
#             image = torch.cat((image,image,image,image), 0) #concatenating images 4 times is not necessary for LIDC dataset, but for MRI we concatenated all of them (flair, f1, f2, pd). This is for reference! :D
#             label = out[random.randint(1, 4)]
#             label = torch.unsqueeze(label, 0)
  
            
            
#             return (image, label, path)
#         else:

#             image = out[0]
#             image = torch.unsqueeze(image, 0)
#             image = torch.cat((image,image,image,image), 0)
#             label = out[random.randint(1, 4)]
#             label = torch.unsqueeze(label, 0)
#             return (image, label)

#     def __len__(self):
#         return len(self.database)


import os
import pickle
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from transform import TransformDataset

class LIDCDataset(Dataset):

    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
        mask_type: str = "random",  # "random", "ensemble", "multi"
        test_flag: bool = True,
    ) -> None:
        super().__init__()
        
        self.data_dir = data_dir
        self.mask_type = mask_type
        self.test_flag = test_flag
        self.masks_per_image = 4
        self.data = {"images": [], "masks": [], "paths": []}
        
        if train_val_test_dir:
            self.load_data(f"{self.data_dir}/{train_val_test_dir}.pickle")
        else:
            self.load_data(f"{self.data_dir}/Train.pickle")
            self.load_data(f"{self.data_dir}/Val.pickle")

    def load_data(self, datafile_path: str):
        max_bytes = 2**31 - 1
        bytes_in = bytearray(0)
        input_size = os.path.getsize(datafile_path)
        with open(datafile_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        new_data = pickle.loads(bytes_in)
        self.data["images"].extend(new_data["images"])
        self.data["masks"].extend(new_data["masks"])
        # add dummy paths for compatibility
        for i in range(len(new_data["images"])):
            self.data["paths"].append(f"{datafile_path}_img_{i}")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, index):
        image = self.data["images"][index].astype(np.float32)
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())

        # Make torch tensor and expand channel to 4 (simulate 4 channels like MRI modalities)
        image_tensor = torch.tensor(image).unsqueeze(0)  # (1, H, W)
        image_tensor = image_tensor.repeat(4, 1, 1)      # (4, H, W)

        # Handle masks
        masks = self.data["masks"][index]

        if self.mask_type == "random":
            mask = masks[random.randint(0, self.masks_per_image - 1)]
        elif self.mask_type == "ensemble":
            mask = np.stack(masks, axis=-1).mean(axis=-1)
            mask = (mask > 0.5).astype(np.uint8)
        elif self.mask_type == "multi":
            mask = masks
        else:
            raise ValueError("Invalid mask_type. Choose from: random, ensemble, multi.")

        # Convert mask to tensor
        if self.mask_type == "multi":
            mask_tensor = torch.tensor(np.stack(mask, axis=0).astype(np.uint8))  # shape: (4, H, W)
        else:
            mask_tensor = torch.tensor(mask.astype(np.uint8)).unsqueeze(0)  # (1, H, W)

        # For compatibility with previous code: return image, label, path (optional)
        path = self.data["paths"][index]
        if self.test_flag:
            return image_tensor, mask_tensor, path
        else:
            return image_tensor, mask_tensor


def get_lidc_dataset(args, mode: str = ""):
    train_dataset = val_dataset = None
    
    if mode == "train":
        train_dataset = LIDCDataset(
            data_dir=args.data_dir,
            train_val_test_dir="Train",
            mask_type=args.mask_type,
            test_flag=False
        )
        train_dataset = TransformDataset(train_dataset, image_size=args.image_size)
    
    if mode == "val":
        val_dataset = LIDCDataset(
            data_dir=args.data_dir,
            train_val_test_dir="Val",
            mask_type=args.mask_type,
            test_flag=True
        )
        val_dataset = TransformDataset(val_dataset, image_size=args.image_size)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    dataset = LIDCDataset(data_dir='data/lidc', train_val_test_dir="Train", mask_type="multi", test_flag=True)
    print("Dataset size:", len(dataset))

    id = random.randint(0, len(dataset) - 1)
    image, masks, path = dataset[id]
    print("Sample path:", path)
    print("Image shape:", image.shape)
    print("Mask shape:", masks.shape)

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(image[0], cmap='gray')
    axs[0, 0].set_title('Image')
    axs[0, 0].axis("off")

    for i in range(3):
        axs[0, i+1].imshow(masks[i], cmap='gray')
        axs[0, i+1].set_title(f'Mask_{i}')
        axs[0, i+1].axis("off")

    mask_e = masks.float().mean(dim=0)
    mask_var = masks.float().var(dim=0)

    axs[1, 0].imshow(mask_e, cmap='gray')
    axs[1, 0].set_title('Mask Ensemble')
    axs[1, 0].axis("off")

    axs[1, 1].imshow(mask_var, cmap='gray')
    axs[1, 1].set_title('Mask Variance')
    axs[1, 1].axis("off")

    sns.heatmap(mask_var.numpy(), ax=axs[1, 2])
    axs[1, 2].set_title('Variance Heatmap')
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()
