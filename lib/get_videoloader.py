from torch.utils.data import DataLoader
from lib.datasets.mixed_dataset import MixedVidDataset
from lib.datasets.video_dataset import VideoDataset
from lib.core.data_loader import CheckpointDataLoader

def get_dataloaders(cfg=None):

    train_bs = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.NUM_WORKERS
    crop_size = cfg.IMG_RES
    dataset_list = cfg.DATASET.LIST
    seqlen = cfg.DATASET.SEQ_LEN
    stride = cfg.DATASET.STRIDE
    valid_set = cfg.DATASET.TEST
    partition = cfg.DATASET.PARTITION

    print('Num of data loading workers:', num_workers)
    print('Sequence length:', seqlen)
    print('Sequence stride:', stride)

    print('Datasets:', dataset_list)
    print('Partition:', partition)

    train = MixedVidDataset(dataset_list, partition, is_train=True, use_augmentation=True, 
                            normalization=True, cropped=True, crop_size=crop_size, 
                            seqlen=seqlen, stride=stride)
    train_loader = CheckpointDataLoader(train, shuffle=True, batch_size=train_bs, num_workers=num_workers)

    test = VideoDataset(valid_set, is_train=False, use_augmentation=False, 
                    normalization=True, cropped=True, crop_size=crop_size, seqlen=16, stride=16)
    test_loader = DataLoader(test, batch_size=8, shuffle=False, num_workers=num_workers)

    return [train_loader, test_loader]


