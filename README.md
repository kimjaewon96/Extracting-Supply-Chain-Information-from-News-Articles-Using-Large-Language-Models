# Extracting Supply Chain Information from News Articles Using Large Language Models: A Fully Automatic Approach

## Installation

This codebase is only tested on Windows 11.

1. Download [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Open `cmd` from directory
3. Type `uv sync`
4. Run each jupyter notebook, using the created virtualenv

## Requirements

### Relation Classification Dataset

To create 'ManualDataset' and 'ManualReducedDataset', the dataset of [Wichmann et al.](https://www.tandfonline.com/doi/full/10.1080/00207543.2020.1720925) is needed. Go to https://github.com/pwichmann/supply_chain_mining and follow the instructions on obtaining the dataset.

### Huggingface Token

To use LLama-3-8B-Instruct, you need an access token from huggingface and the permission to use the model from Meta. Here are the necessary steps:

1. Go to https://huggingface.co, and create an account.
2. Then, go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct, and acquire access to the model. This may take several days.
3. If access to the model is granted, go to https://huggingface.co/settings/tokens and create a new access token. Select 'Read' from 'Token type'.
4. The access token will be shown only once after it is created (starting with 'hf_'). Copy and paste it in the .env file as 'ACCESS_TOKEN', such as `ACCESS_TOKEN=hf_...`

### Case Study Dataset

To obtain the case study dataset (mining.json, mining_processed.json), please [contact the author](pokedexter@korea.ac.kr).