from pretrain_ssl.augment import MultiCropAugmentation
from pretrain_ssl.dataset import AmazonSSLDataset, multicrop_collate_fn
from torch.utils.data import DataLoader

print("Starting data test...")

aug = MultiCropAugmentation(debug=True)
ds = AmazonSSLDataset(data_path="./dataset", transforms=aug, debug=True)
dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=multicrop_collate_fn)

batch = next(iter(dl))
print("Batch loaded")
print("Crop batch shapes:", [x.shape for x in batch["crops"]])
