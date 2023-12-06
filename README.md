# Image Processing and Augmentation Tool

## Description

This project is an image processing and augmentation tool designed to apply various transformations to images. Data augmentation in the context of machine learning and image processing is a technique used to enhance the size and quality of training datasets by creating modified versions of the data. This process helps in improving the robustness and effectiveness of models, especially in tasks like image recognition and classification.

## Features

- Apply multiple image augmentation techniques like flipping, brightness enhancement, contrast enhancement, sharpening, edge enhancement, gamma correction, and equalization.
- Ability to load processing parameters from a JSON file.
- Process images in bulk from a specified input directory.

---

## Getting Started

### Requirements

- Python 3.x
- Pip package manager

### Setup and Installation

1. **Clone the Repository**

```bash
git clone https://github.com/renan-siqueira/python-data-augmentation-tool.git
cd python-data-augmentation-tool
```

2. **Create and Activate a Virtual Environment (Optional but recommended)**

- For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

- For Unix or MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Required Dependencies**

```bash
pip install -r requirements.txt
```

---

## How to Use

1. Place your images in the input directory specified in the `settings/config.py` file.

2. Modify the `json/params.json` file to set your desired augmentation parameters.

3. Run the main script to process the images:

```bash
python main.py
```

4. Processed images will be saved in the output directory specified in the `settings/config.py` file.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions to this project are welcome. Please adhere to this project's Code of Conduct.

---

## Authors

- Renan Siqueira Antonio
