# Pneumothorax Classification

## Overview
This repository contains code and resources for classifying pneumothorax conditions from medical images. The project leverages deep learning techniques to analyze and predict the presence of pneumothorax in chest X-ray images.

## Prerequisites
- Docker
- Internet connection for downloading model weights

## Dataset 
- train and test image datasets [abhishek/siim-png-images](https://www.kaggle.com/datasets/abhishek/siim-png-images)
- Mapping csv for training and testing [chogerlate/siim-train-dataframe](https://www.kaggle.com/datasets/chogerlate/siim-train-dataframe)

## Weights Downloading
Before building the Docker image, you need to download the pre-trained model weights. Follow these steps:

1. Create a directory named `weights` in the root of the repository:
    ```sh
    mkdir weights
    ```
2. Install `gdown` to download model weight from Google Drive
   ```sh
   pip install gdown
   ```
4. Download the weights file from the provided link and place it in the `weights` directory. 
    ```sh
    gdown https://drive.google.com/uc?id=<file-id>
    ```
   **Note** that the URL might change in the future; if it is, you can DM me.
5. Copy downloaded weight into weights directory.
## Building the Docker Image
To build the Docker image, navigate to the root of the repository and run the following command:
```sh
docker build -t pneumothorax-classification .
```

## Running the Docker Container
Once the Docker image is built, you can run the container using the following command:
```sh
docker run -p 80:80 pneumothorax-classification
```

This will start the application, and it will be accessible at `http://localhost:80`.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [Dataset Source](https://www.kaggle.com/datasets/abhishek/siim-png-images?select=test_png)
- [Pre-trained Model --> DM ME!](https://www.instagram.com/chogerlatte/)

For more details, refer to the documentation or contact the project maintainers.
