import os
import json
import zipfile
from Storage.storageEnum import FileEnum


class StorageController:
    """
    A class used to remove and create the directories and files used in the execution

    Methods
    -------
    compress_files(files, compressed_file_path)
        Create a compressed folders with the files selected
    extract_compressed_files(source_path, destination_path, file_extension=".h5")
        Extract the files compressed in the source path
    remove_files_from_list(files)
        Delete the files selected
    remove_files_from_folder(directory_path, file_extension=".h5")
        Delete the files from the folder selected
    create_directory(path)
        Create a directory
    create_json_file(path, content)
        Create a json file with the content selected
    """

    def __init__(self):
        pass

    @staticmethod
    def compress_files(files, compressed_file_path):
        """Create a compressed folders with the files selected

        Parameters
        ----------
        files : list
            List of files
        compressed_file_path : string
            Path to the file compressed to create
        """
        zip_obj = zipfile.ZipFile(compressed_file_path, 'w')

        for file in files:
            file_path = file[FileEnum.FILE_PATH.value]
            file_name = file[FileEnum.FILE_NAME.value]
            zip_obj.write(file_path + file_name, file_name, compress_type=zipfile.ZIP_DEFLATED)

        zip_obj.close()

    @staticmethod
    def extract_compressed_files(source_path, destination_path, file_extension=".h5"):
        """Extract the files compressed in the source path

        Parameters
        ----------
        source_path : string
            Path of the source file to extract
        destination_path : string
            Path of the destination file to extract
        file_extension: string, optional
            Extantion of the file to extract (Default is ".h5")
        
        Returns
        -------
        array
            Return an array with all the extracted file's path
        """

        with zipfile.ZipFile(source_path, "r") as zip_ref:
            zip_ref.extractall(destination_path)

        files = []
        for file in os.listdir(destination_path):
            if not file.endswith(file_extension):
                continue
            files.append(os.path.join(destination_path, file))

        return files

    @staticmethod
    def remove_files_from_list(files):
        """Delete the files selected

        Parameters
        ----------
        files : array
            Array of the files' source path to remove 
        """
        for file in files:
            file_path = file[FileEnum.FILE_PATH.value]
            file_name = file[FileEnum.FILE_NAME.value]
            os.remove(file_path + file_name)

    @staticmethod
    def remove_files_from_folder(directory_path, file_extension=".h5"):
        """Delete the files from the folder selected

        Parameters
        ----------
        directory_path : string
            Path of the directory containing the files to remove 
        file_extension: string, optional
            Extantion of the files to remove (Default is ".h5")
        """
        for file in os.listdir(directory_path):
            if not file.endswith(file_extension):
                continue
            os.remove(os.path.join(directory_path, file))

        os.rmdir(directory_path)

    @staticmethod
    def create_directory(path):
        """Create a directory

        Parameters
        ----------
        path : string
            Path of the directory to create
        """
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def create_json_file(path, content):
        """Create a json file with the content selected

        Parameters
        ----------
        path : string
            Path of the directory to create
        content : dictionary
            Content of the file to create
        """
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(content, f)
