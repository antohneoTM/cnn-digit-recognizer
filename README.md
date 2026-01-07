# Convolutional Neural Network Digit Recognizer
Developed By: Antonio Camacho\
Email: camacho.ant.626@gmail.com\
LinkedIn: <a href="https://www.linkedin.com/in/antcamacho" target="_blank">/in/antcamacho</a>

## About
Example of Convolutional Neural Network (CNN) written in Python, featuring a fully trained PyTorch AI model.
<a href="https://en.wikipedia.org/wiki/Convolutional_neural_network" target="_blank">More about CNNs</a>

### MNIST
This model was trained using images from the MNIST database (Modified National Institute of Standards and Technology).
The database features thousands of normalized grayscaled images with a resolution of 28x28. Being one of the most popular digit databases with various handwritten digits the MNIST database was a perfect fit for the model.
While the database doesn't contain high resolution images of real world digits, each image is clear and easily identified and throughly labeled.

<a href="https://en.wikipedia.org/wiki/MNIST_database" target="_blank">More about MNIST database</a>

### PyTorch
PyTorch is a very popular machine learning library used for powerful deep learning researched and training Artifical Intelligence models for various programming languages. (But most commonly Python)
Quickly rising to the industry standard for machine learning in 2026 it was the ideal option for this project.

You can learn more about PyTorch and the PyTorch foundation <a href="https://pytorch.org" target="_blank">here</a>

## Instructions
### Requirements
- Project files avaiable on this repo:\

    `git clone git@github.com:antohneoTM/cnn-digit-recognizer.git`

- Python 3.10+

    To install the latest version of Python go to:\
    <a href="https://www.python.org/downloads" target="_blank">python.org/downloads</a>

- pip Package Manager:

    The standard package manager for Python
    For more information about pip go to:\
    <a href="https://www.pip.pypa.io/en/stable/" target="_blank">pip.pypa.io/en/stable/</a>

- Python Packages (pip package manager):

    - torch 2.8.0

    - torchvision 0.23.0

    To install these onto your Python enviroment run:\
    `pip install -r requirements.txt`\
    - requirements.txt includes more detailed information regarding the installed packages

    or:\
    `pip install torch`\
    `pip install torchvision`

###How to run
1. Open the project's root directory in a terminal
2. Active your Python environment
3. Make sure Python packages are installed and check for compatible version\
    `pip list`

    Should give an output like:\
    ```
    Package   Version
    --------  -------
    filelock  3.20.0
    fsspec    2025.9.0
    Jinja2    3.1.6
    ...
    ...
    ```
4. In the project's root directory run:\
    `python3 code/main.py`
5. Train some digit recognizer CNNs!

### Notes
- There is an included pre-trained CNN model included with the project with an 99.54% accuracy

- Training a new CNN model can take up to 30 minutes on high end systems. Systems with Nvidia graphics cards tend to see higher average performance on PyTorch.
- After training a new CNN model, the existing model will be replaced with the newly trained model at:\
    `model/digit-recognizer.pth`

- There are included sample images that are used when selecting `[P]redict` when running `main.py`. These include 20 black and white and written digits for each digit.

- You can add your own images to the folder `sample_images`. Remember grayscale images, digits drawn clearly in the center of the image are best, and 'square' images work best.
- You can add images that don't meet these preferences, but results will vary.
- The digit `8` has the most varying results as the model tends to mistake `3` and `6` for it fairly often.
