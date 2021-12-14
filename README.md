# AI_APPs_Flower_Classification
Use Tensorflow to classify an Image database, and from Jupyter notebook translate to command line parser option to make an Image Classifier command line app, whose input is an image, and output is the few classified names in the order of likelihood.

Command line to run the apps:
$ python predict.py ./test_images/cautleya_spicata.jpg image_classifier_model_udacity_project_1633673482.h5 --top_k 5 --category_names label_map.json

Specifications

predict.py file that uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Create a module just for utility functions like preprocessing images. Make sure to include all files necessary to run the predict.py file in your submission.

The predict.py module should predict the top flower names from an image along with their corresponding probabilities.

Basic usage:

$ python predict.py /path/to/image saved_model

Options:

    --top_k : Return the top KKK most likely classes:

$ python predict.py /path/to/image saved_model --top_k KKK

    --category_names : Path to a JSON file mapping labels to flower names:

$ python predict.py /path/to/image saved_model --category_names map.json

The best way to get the command line input into the scripts is with the argparse module in the standard library. You can also find a nice tutorial for argparse here.
Examples

For the following examples, we assume we have a file called orchid.jpg in a folder named/test_images/ that contains the image of a flower. We also assume that we have a Keras model saved in a file named my_model.h5.

Basic usage:

$ python predict.py ./test_images/orchid.jpg my_model.h5

Options:

    Return the top 3 most likely classes:

$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3

    Use a label_map.json file to map labels to flower names:

$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json

Workspace
Install TensorFlow

Before you run any commands in the terminal make sure to install TensorFlow 2.0 and TensorFlow Hub using pip as shown below:

$ pip install -q -U "tensorflow-gpu==2.0.0b1"

$ pip install -q -U tensorflow_hub

Images for Testing

Here are 4 images in the ./test_images/ folder for you to check your prediction.py module. The 4 images are:

    cautleya_spicata.jpg
    hard-leaved_pocket_orchid.jpg
    orange_dahlia.jpg
    wild_pansy.jpg

