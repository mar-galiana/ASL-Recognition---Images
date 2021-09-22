import numpy as np
from Model.signs import Signs
from Model.modelEnum import Image
from Model.modelEnum import Environment
from Structures.iUtilStructure import Structure
from tensorflow.python.keras.utils import np_utils
from Exception.modelException import EnvironmentException
from Model.DatasetController.inputModel import InputModel
from Model.DatasetController.outputModel import OutputModel
from Exception.structureException import StructureException
from Structures.NeuralNetworks.neuralNetworkEnum import NeuralNetworkTypeEnum


class Model:
    """
    A class used to sync up all the functionalities that refer to the database 

    Attributes
    ----------
    signs : Signs
        a formatted string to print out what the animal says
    output_model : OutputModel
        the name of the animal
    input_model : InputModel
        the sound that the animal makes

    Methods
    -------
    create_pickle(pickle_name, dataset, environments_separated)
        Save the dataset into a pickle
    set_pickles_name(names)
        Set the pickle's name file used when reding them
    get_pickles_name()
        Get the pickle's name file used when reding them
    read_reduced_pickles()
        Read a reduced fraction of the pickle's samples  
    load_image(src, as_gray)
        Load image from sources
    get_x(environment)
        Get the data stored in the pickles
    get_y(environment)
        Get the labels stored in the pickles
    set_x(environment, data)
        Set the data stored in the pickles
    set_y(environment, label)
        Set the labels stored in the pickles
    get_signs_dictionary()
        Get the signs from the pickels selected
    get_sign_values(labels)
        Get the value of all the signs from the pickles selected
    get_sign_value(label)
        Get the value of the sign from the pickles selected
    get_sign_based_on_value(value):
        Get the sign given its value.
    get_categorical_vectors(environment, n_classes)
        Convert data to categorical vectors
    convert_to_one_hot_data()
        Convert data to one hot
    resize_data(structure, environment, shape, nn_type=NeuralNetworkTypeEnum.CNN)
        Resize data depending on the structure
    resize_image(image, structure, nn_type=NeuralNetworkTypeEnum.CNN)
        Resize image depending on the structure
    """

    def __init__(self, width=150, height=None):
        """
        Parameters
        ----------
        width : number
            Width used to resize the images
        height : number
            Height used to resize the images
        """
        self.signs = Signs()
        self.output_model = OutputModel(width, height)
        self.input_model = InputModel()


    #region dataset_controller
    def create_pickle(self, pickle_name, dataset, environments_separated):
        """Save the dataset into a pickle.

        Parameters
        ----------
        pickle_name : string
            Name of the file to create
        dataset : Dataset
            Name of the dataset to store in the pickle
        environments_separated : boolean
            Boolean indicating if the database contains the samples separated in test
            and train folders.
        """
        self.output_model.create_pickle(pickle_name, dataset, environments_separated)

    def set_pickles_name(self, names):
        """Set the pickle's name file used when reding them.

        Parameters
        ----------
        names : array
            Array with the pickle's names
        """
        self.input_model.set_pickles_name(names)

    def get_pickles_name(self):
        """Get the pickle's name file used when reding them.

        Returns
        -------
        list
            Array with the pickle's names
        """
        return self.input_model.get_pickles_name()

    def read_reduced_pickles(self):
        """Read a reduced fraction of the pickle's samples.
        """
        self.input_model.combine_pickles_reducing_size(Environment.TRAIN)
        self.input_model.combine_pickles_reducing_size(Environment.TEST)

    def load_image(self, src, as_gray):
        """Load image from sources.

        Parameters
        ----------
        src : string
            Path of the image to read
        as_gray : boolean
            If True, convert color images to gray-scale
        
        Returns
        -------
        ndarray
            Third dimension image array
        """
        return self.output_model.load_image(src, as_gray)

    def get_x(self, environment):
        """Get the data stored in the pickles.

        Parameters
        ----------
        environment : Environment
            Environment of the data to get
        
        Returns
        -------
        array
            Array with the dataset data
        """
        data = self.input_model.get_data(environment)
        return np.array(data[Image.DATA.value])

    def get_y(self, environment):
        """Get the labels stored in the pickles.

        Parameters
        ----------
        environment : Environment
            Environment of the labels to get
        
        Returns
        -------
        array
            Array with the dataset labels
        """
        data = self.input_model.get_data(environment)
        return np.array(data[Image.LABEL.value])

    def set_x(self, environment, data):
        """Set the data stored in the pickles.

        Parameters
        ----------
        environment : Environment
            Environment of the data to set
        data : array
            Array of data samples
        """
        self.input_model.set_x(environment, data)

    def set_y(self, environment, label):
        """Set the labels stored in the pickles.

        Parameters
        ----------
        environment : Environment
            Environment of the lables to set
        label : array
            Array of labels samples
        """
        self.input_model.set_y(environment, label)
    #endregion

    #region signs_controller 
    def get_signs_dictionary(self):
        """Get the signs from the pickels selected.

        Returns
        -------
        dictionary
            Dictionary of signs
        """
        return self.signs.get_signs_dictionary()

    def get_sign_values(self, labels):
        """Get the value of all the signs from the pickles selected.

        Parameters
        ----------
        label : array
            Array of labels samples

        Returns
        -------
        array
            Array of label's values
        """
        return self.signs.transform_labels_to_sign_values(labels)

    def get_sign_value(self, label):
        """Get the value of the sign from the pickles selected.

        Parameters
        ----------
        label : array
            A label samples

        Returns
        -------
        array
            The value of the input label
        """
        return self.signs.get_sign_value(label)
    
    def get_sign_based_on_value(self, value):
        """Get the sign given its value.

        Parameters
        ----------
        label : array
            A label samples

        Returns
        -------
        array
            The value of the input label
        """
        return self.signs.get_sign_based_on_value(value)

    #endregion

    #region data_processing
    def get_categorical_vectors(self, environment, n_classes):
        """Convert data to categorical vectors.

        Parameters
        ----------
        environment : Environment
            Environment of the data to convert
        n_classes : number
            Number of classes in the dataset

        Returns
        -------
        array
            Array of the data converted to categorical vectors 
        """
        if not isinstance(environment, Environment):
            raise EnvironmentException("Environment used is not a valid one")

        vectors = self.get_sign_values(self.get_y(environment))
        y_data = np_utils.to_categorical(vectors, num_classes=n_classes)
        return y_data

    def convert_to_one_hot_data(self):
        """Convert data to one hot.

        Returns
        -------
        number
            Number of classes in the dataset
        """
        x_train = self.get_x(Environment.TRAIN).astype('float32')
        x_test = self.get_x(Environment.TEST).astype('float32')

        # normalizing the data to help with the training
        x_train /= 255
        x_test /= 255

        # one-hot encoding
        n_classes = np.unique(self.get_y(Environment.TRAIN)).shape[0] + 1
        y_train = self.get_categorical_vectors(Environment.TRAIN, n_classes)
        y_test = self.get_categorical_vectors(Environment.TEST, n_classes)

        self.set_y(Environment.TRAIN, y_train)
        self.set_y(Environment.TEST, y_test)

        return n_classes

    def resize_data(self, structure, environment, nn_type=NeuralNetworkTypeEnum.CNN):
        """Resize data depending on the structure.

        Parameters
        ----------
        structure : Structure
            Structure that the data is gone be used to
        environment : Environment
            Environment of the data to convert            
        nn_type : NeuralNetworkTypeEnum, optional
            Neural network type (default is CNN)

        """
        # Input control
        if not isinstance(structure, Structure):
            raise StructureException("Incorrect structure exception")

        if not isinstance(nn_type, NeuralNetworkTypeEnum):
            raise StructureException("Incorrect neural network type")

        # Check structure to select the resize formt
        shape = self.get_x(environment).shape
        data = self.get_x(environment)

        if nn_type == NeuralNetworkTypeEnum.ANN or structure == Structure.DecisionTree:
            resized_data = self.__resize_ann_and_dt_data(data, shape)

        else:
            resized_data = self.__resize_cnn_data(data, shape)
        
        self.set_x(environment, resized_data)
    
    def resize_image(self, image, structure, nn_type=NeuralNetworkTypeEnum.CNN):
        """Resize image depending on the structure.

        Parameters
        ----------
        image : array
            Pixels of the image to resize 
        structure : Structure
            Structure that the data is gone be used to   
        nn_type : NeuralNetworkTypeEnum, optional
            Neural network type (default is CNN)
        
        Returns
        -------
        array
            array with a single element, the image resized
        """
        # Input control
        if not isinstance(structure, Structure):
            raise StructureException("Incorrect structure exception")
        
        data = np.array([image])
        shape = data.shape

        if nn_type == NeuralNetworkTypeEnum.ANN or structure == Structure.DecisionTree:
            resized_data = self.__resize_ann_and_dt_data(data, shape)

        else:
            resized_data = self.__resize_cnn_data(data, shape)

        return resized_data

    def __resize_cnn_data(self, data, shape):
        return data.reshape(shape[0], shape[1], shape[2], 1)
        
    def __resize_ann_and_dt_data(self, data, shape):
        return data.reshape(shape[0], shape[1]*shape[2])

    #endregion