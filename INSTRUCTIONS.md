# Training Macaque Identification Model

This document provides instructions on how to train the identification model for macaques using the GorillaVision system.

## Prerequisites

1. Make sure Docker is installed and properly configured with GPU support
2. Ensure the `gorilla_triplet` Docker image has been built successfully with `docker build -t gorilla_triplet .`
3. Verify the Gibraltar_Macaques dataset is in the correct location (should be in the parent directory of this repository)

## Dataset Structure

The Gibraltar_Macaques dataset should be organized with one directory per individual, containing images of that individual:

```
Gibraltar_Macaques/
├── Abby/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Adele/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Running the Training Pipeline

1. The training pipeline is fully automated in the bash script `gorillavision/reid-system/scripts/train_macaque_identification.sh`.

2. Execute the script from the root directory of the repository:

   ```bash
   cd ~/FacialRecognitionTest
   ./gorillavision/reid-system/scripts/train_macaque_identification.sh
   ```

3. The script will:
   - Split your dataset into train/database/evaluation sets
   - Train the identification model
   - Create an identification database
   - Evaluate the model performance

4. All trained models and the identification database will be stored in the `macaque_models` directory.

## Pipeline Details

The training process consists of these steps:

1. **Data Preparation**: The script splits your dataset into three parts:
   - Training set (70% of images per individual)
   - Database set (15% of images per individual)
   - Evaluation set (15% of images per individual)

2. **Model Training**: Runs the `train_identification.py` script inside the Docker container using the ViT (Vision Transformer) backbone with triplet loss.

3. **Database Creation**: Creates a database of known identities using the trained model.

4. **Model Evaluation**: Evaluates the model performance on the test set.

## Configuration

If you need to modify the training parameters:

1. Edit the configuration file at:
   `gorillavision/reid-system/gorillavision/configs/custom/macaque_config.json`

2. You can adjust:
   - Image input size
   - Batch size
   - Learning rate
   - Number of epochs
   - Augmentation settings

## Troubleshooting

- If the training fails, check the error messages in the console output.
- Ensure GPU is properly accessible to Docker (run `nvidia-smi` to verify).
- If memory issues occur, try reducing the batch size in the configuration file.
- Make sure the dataset has enough images per individual (minimum 3 per individual is required). 