# PROCESAMIENTO DEL LENGUAJE DE SIGNOS Y SU CONVERSIÓN A TEXTO

##OBJECTIVE
In this project we wanted to implement a predictive model that uses images as an input data. These images refer to thirty-eight static gestures, belonging to the American Sign Language (ASL). The gestures represent the English alphabet and the numbers from zero to ten.

There have been two predictive models used: decision trees and neural networks. The Extreme Gradient Boosting algorithm was used with the decision tree, obtening an accuracy of 100%. It was eventually discarded because, as the number of samples in the dataset increased, its time cost increased too much to be used under this project execution conditions.

As far as neural networks are concerned, two types were implemented: one artificial and one convolutional. Two types of classifier were used: multi-class and binary, using the One vs All strategy.

The best results were obtained with the multi-class convolutional neural net- work, achieving an accuracy of 99.52 % with a total execution time of 1 hour, 55 minutes and 36 seconds, and a memory cost of 12,88 GB.

##IMPLEMENTATION
This project was implemented using the Python 3.9 interpreter and the open source package conda.

##STEPS TO EXECUTE IT
In order to run it, you must first install the requirements from the requirements.txt file, using the command:
``pip install -r requirements.txt``

There are some packages that are not included in this file, like the graphviz, it can be installed using the command:
``conda install graphviz python-graphviz``

Once the requirements are installed, you need to execute the first strategy, called: setup. In case that you don't know how to execute it, you can use the help strategy. Entering the arguments: 
``--help``

The setup strategy will install all the folders and files needed to execute the other strategies.

Then you need to clone the repository stored at the URL https://github.com/marGaliana/SignLanguageProcessingDataset.git. it's very important to clone it in the Assets/Dataset/Images path. These folders will be created after the setup strategy has been executed. This respository contains the samples from each dataset used.

To execute the other strategies it's recommended to check the information shown in the help strategy.

If when executing the project it doesn't find the files that are correctly located, change the value of the variable ASSETS stored in the Src/Constraints/path.py file. It's value will depend on the environment where the execution is done.

In case there is any doubt of how to start executing the project, the file SignGestureDetection/SignLanguageProcessing.ipynb contains an exemple of how to execute each one of these strategies, this file can be opend with the jupyter notebook. It'srecomended to execute the prroject in this environment.