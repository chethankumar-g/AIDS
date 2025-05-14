
# Automated Infection Detection System (AIDS)

An Automated Infection Detection System that utilizes advanced deep learning algorithms, such as Convolutional Neural Networks (CNN), to analyze chest X-ray images and detect Pneumonia. By identifying patterns associated with Pneumonia, the system provides rapid and accurate diagnoses, assisting healthcare professionals and improving efficiency in clinical settings.

## Features

- **Deep Learning Integration**: Employs CNN models for precise image analysis.
- **Rapid Diagnosis**: Quickly processes chest X-rays to identify signs of Pneumonia.
- **Clinical Support**: Aids healthcare professionals in making informed decisions.
- **User-Friendly Interface**: Simplified interaction for ease of use.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/chethankumar-g/AIDS.git
   cd AIDS
   ```

2. **Install Dependencies**:

   Ensure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run Colab Notebooks**:

   - Open the Colab notebooks provided in the repository.
   - Mount your Google Drive and load the dataset.
   - Train the models using the selected CNN architectures (e.g., VGG16, ResNet, Inception).
   - The trained models should be saved to the `/models` directory.

2. **Main Application**:

   The `main.py` script integrates preprocessing and model prediction for streamlined usage.

   ```bash
   python main.py
   ```

## Project Structure

- `main.py`: Main application script integrating preprocessing and prediction.
- `model_loading.py`: Handles loading of the trained CNN model.
- `preprocess.py`: Preprocesses chest X-ray images for analysis.
- `requirements.txt`: Lists all Python dependencies.
- `notebooks/`: Colab notebooks for training CNN models.
- `models/`: Directory containing trained CNN models.
- `templates/`: Contains HTML templates for the web interface.
- `static/uploads/`: Directory for storing uploaded images.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Developed by [chethankumar-g](https://github.com/chethankumar-g)
- Inspired by advancements in medical imaging and deep learning.
