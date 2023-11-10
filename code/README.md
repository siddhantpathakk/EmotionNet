# Emotion Detection in Conversational Speech

## Project Overview

In this project, we introduce an innovative approach to emotion detection in conversational speech. Utilizing a multi-modal framework that integrates both audio and textual data, our system is capable of discerning emotions within multi-speaker dialogues. This is achieved through the extraction of audio features using pre-trained models and processing of text embeddings, coupled with an attentive bi-directional GRU network. This network dynamically captures the context and the inter-speaker emotional influences.

## Key Features

- Multi-modal emotion recognition leveraging audio and text data.
- Utilization of pre-trained models for robust audio feature extraction.
- Implementation of an attentive bi-directional GRU for contextual understanding.
- Evaluation on the MELD dataset, demonstrating effective emotion detection.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- Pip (Python Package Installer)

### Installation

1. Clone the repository
   `git clone https://github.com/siddhantpathakk/emotion-rnn.git`
2. Navigate to the project directory:
   `cd emotion-rnn`
3. Install the required packages (preferably using Conda/Miniconda):
   `conda env myenv --file environment.yml`

### Usage

To see the training notebook, please refer to `./audio/train.ipynb`

## Dataset

This project uses the MELD conversational dataset for training and testing. Ensure that you have the dataset downloaded and placed in the `audio` directory. We used preprocessed features available [here](https://github.com/declare-lab/conv-emotion.).

## Structure

* `audio/`: Contains audio data and extracted features.
* `src/`: Source code for the project including model definitions and training scripts.
  * `inference.py`: Script for performing inference.
  * `model.py`: Defines the bi-directional GRU model.
  * `dataloader.py`: Code for loading and preprocessing data.
  * `attention.py`: Implementation of the attention mechanisms.
* `train.ipynb`: Jupyter notebook for training the model.
* `infer.ipynb`: Jupyter notebook for using the model for
* `exploratory_data_analysis.ipynb`: Jupyter notebook for performing standard EDA about the MELD dataset.
* `environment.yml`: Conda environment file for setting up the Python environment.
* `LICENSE`: The license under which this project is distributed.
* `README.md`: This file, describing the project and how to use it.

## Contributing

Please read [CONTRIBUTING.md]() for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Siddhant Pathak** - *Initial work* - [@siddhantpathakk](https://github.com/siddhantpathakk)

## License

This project is licensed under the MIT License - see the [LICENSE](https://chat.openai.com/c/LICENSE) file for details.

## Conclusion

Our methodology opens new avenues in speech emotion recognition by focusing on the nuances of conversational context and speaker interactions. By implementing dual attention mechanisms and a bi-directional GRU, our system adeptly identifies emotional cues
