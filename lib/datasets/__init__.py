from .dataset_h5py import Dataset, ValDataset, TestDataset, CanonicalDataset
from torch.utils.data import DataLoader

def find_dataset_using_name(name):
    mapping = {
        "Video": Dataset,
        "VideoVal": ValDataset,
        "VideoTest": TestDataset,
        "VideoNovelPose": TestDataset,
        "VideoNovelView": TestDataset,
        "Canonical": CanonicalDataset,
    }
    cls = mapping.get(name, None)
    if cls is None:
        raise ValueError(f"Fail to find dataset {name}") 
    return cls


def create_dataset(metainfo, split):
    dataset_cls = find_dataset_using_name(split.type)
    dataset = dataset_cls(metainfo, split)
    return DataLoader(
        dataset,
        batch_size=split.batch_size,
        drop_last=split.drop_last,
        shuffle=split.shuffle,
        num_workers=split.worker,
        persistent_workers=True,
        pin_memory=True
    )