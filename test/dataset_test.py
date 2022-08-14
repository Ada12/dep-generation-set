import os

from torch.utils.data import DataLoader

import constants
from my_dataset.json_dataset import JsonDataset


def test_JsonDataset():
    dataset = JsonDataset(source_path=os.path.join(constants.INPUT_DIR, 'train_x_list.json'),
                          target_path=os.path.join(constants.INPUT_DIR, 'train_y_list.json'),
                          vocab_path=os.path.join(constants.INPUT_DIR, 'dep2id.json'))
    loader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    for sourve_vec, target_vec in loader:
        print(sourve_vec)
        print(target_vec)
