# CVH-Rec: A novel method for web API recommendation based on the cross-view HGNNs

This repository contains the dataset and source code for the paper:\
**"CVH-Rec: A novel method for web API recommendation based on the cross-view HGNNs"**

## 1. CVH-Rec

### 1.1 Environment Setup

Our code has been tested under **Python 3.12.3**, and the following packages are required:

    h5py==3.11.0 
    scikit_learn==1.4.2 
    sentence_transformers==2.7.0 
    torch==2.7.0+cu126

We recommend setting up the environment using **Conda** to ensure compatibility and ease of installation.

#### ✅ Step 1: Create and activate a new Conda environment

    conda create -n CVH-Rec python=3.12.3 -y
    conda activate CVH-Rec 
    
#### ✅ Step 2: Install PyTorch manually

Visit the official PyTorch installation page [here](https://pytorch.org/) to choose the correct version for your environment.\
For example, to install **PyTorch with CUDA 12.6**, use:

    pip install torch --index-url https://download.pytorch.org/whl/cu126

#### ✅ Step 3: Install the required Python packages

Install all dependencies (except PyTorch) via `requirements.txt`:

    pip install -r requirements.txt

> ⚠️ Note: The `requirements.txt` file includes all necessary libraries **except for PyTorch**, which should be installed separately to match your system and CUDA version.


### 1.2 Usage

#### ✅ Step 1: Data generation (Optional)

> This step can be skipped, as all required training and testing files have already been generated in advance. However, if you intend to apply the method to a new dataset, you may execute the following script.

To generate the dataset, start Python in **command line** mode and run:

    python generate_dataset.py

This script uses the original files `Original Dataset/mashups.json` and `Original Dataset/apis.json` as input and performs the following operations:

*   Generates **training set, testing set, and validation set**  under the `dataset` folder.

*   Creates a `data` folder containing:

    *   `API_vectors.h5` and `vectors.h5`: Semantic embedding vectors for API and mashup descriptions.

    *   `api_tag_vector.h5` and `mashup_tag_vector.h5`: Semantic embedding vectors for API and mashup tags.

Each directory under the `dataset` folder contains the following files:

*   `RS.csv`: Test set of mashup–API pairs.

*   `TE.csv`: Training set of mashup–API pairs.

*   `VA.csv`: Validation set of mashup–API pairs.

*   `api_tags.csv`, `Api_tag_mapping.csv`: API–tag relationships and tag–index mappings.

*   `mashup_tags.csv`, `mashup_tag_mapping.csv`: Mashup–tag relationships and tag–index mappings.

#### ✅ Step 2: **Model Training and Testing and Evaluation metrics**

The provided wrapper script `run.py` is designed to **automate the entire experimental pipeline**, including:

*   Sequential training and testing on the dataset

*   Printing **evaluation metrics** after obtaining the recommendation results for the testing mashups.

✅ **If you use the Conda, ensure that the Conda environment is activated before running the script:**

    conda activate CVH-Rec

To train and test the model, start Python in **command line** mode and execute the following (in one line):

    python run.py

After execution, the results will be saved in the `output` folder, which includes recommendation results for the testing mashups.

After training and testing process is completed, the corresponding evaluation metrics will be printed to the terminal in the following format:

    The performance of CVH-Rec is as follows: 
    N=3 -> Precision: 0.3394, Recall: 0.6302, MAP: 0.7313, NDCG: 0.6515
    N=5 -> Precision: 0.2310, Recall: 0.6877, MAP: 0.7291, NDCG: 0.6660
    N=10 -> Precision: 0.1322, Recall: 0.7565, MAP: 0.7174, NDCG: 0.6890
    N=20 -> Precision: 0.0737, Recall: 0.8141, MAP: 0.6956, NDCG: 0.7072

### 1.3 Description of Essential Folders and Files

| Name          | Type   | Description                                                                                                                                                                                                                                                                                       |
| ------------- | ------ |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data          | Folder | Data files required for the experiment. Specifically: **api\_tag\_vector.h5** stores the semantic vectors of the API tags, **vector.h5** stores the semantic vectors of mashup descriptions, **api\_vector.h5** stores the semantic vectors of API descriptions, **mashup\_tag.h5** stores the semantic vectors of mashup tags |
| Original Dataset | Folder | Save the data related to mashups and APIs used in the experiment, including the invocation relationships between mashups and APIs, the descriptions and tags of mashups, and the descriptions and tags of APIs.                                                                                   |
| main.py       | File   | Model training and testing python file of CVH-Rec                                                                                                                                                                                                                                                 |
| run.py        | File   | A **wrapper script** that automates the entire process of training and testing the CVH-Rec model under                                                                                                                                                                                            |
| Models.py     | File   | Model modules of CVH-Rec                                                                                                                                                                                                                                                                          |
| utility       | Folder | Tools and essential libraries used by CVH-Rec                                                                                                                                                                                                                                                     |

####

