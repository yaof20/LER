# LER
Dataset and source code for AAAI 2023 paper "[Unsupervised Legal Evidence Retrieval via Contrastive Learning with Approximated Positive](https://ojs.aaai.org/index.php/AAAI/article/view/25603)"


## Overview

### Dataset

The full dataset can be downloaded via [Google Drive](https://drive.google.com/drive/folders/1JL5QeeUSyncUGHhUPvX-vPIySPmF6JSN?usp=sharing). You can also check the `data_sample` folder for a glimpse of our dataset.

### Code
- Prepare the dataset
  - Unzip this file to get `/data` folder and put it in the root path (`LER`)
  - Leave the file names unchanged (consistent with the config files in `LER/config` folder)
  - The directory should look like this:
    ```
    ├── config
    │   ├── ...
    │   └── ...
    ├── data
    │   ├── test
    │   │   ├── test_dev-set-200.json
    │   │   └── test_test-set-719.json
    │   └── train
    │       └── train_all_record-wo-test.jsonl
    ...
    ```
- Check the `config` folder for different experiment settings. 
- We will try to add more detailed explanations soon ...
