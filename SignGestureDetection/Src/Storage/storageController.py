import os
import zipfile
from Storage.storageEnum import FileEnum


class StorageController:

    def __init__(self):
        pass

    @staticmethod
    def compress_files(files, compressed_file_path):

        zip_obj = zipfile.ZipFile(compressed_file_path, 'w')

        for file in files:
            file_path = file[FileEnum.FILE_PATH.value]
            file_name = file[FileEnum.FILE_NAME.value]
            zip_obj.write(file_path + file_name, file_name, compress_type=zipfile.ZIP_DEFLATED)

        zip_obj.close()

    @staticmethod
    def extract_compressed_files(source_path, destination_path, file_extension=".h5"):

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
        for file in files:
            file_path = file[FileEnum.FILE_PATH.value]
            file_name = file[FileEnum.FILE_NAME.value]
            os.remove(file_path + file_name)

    @staticmethod
    def remove_files_from_folder(directory_path, file_extension=".h5"):
        for file in os.listdir(directory_path):
            if not file.endswith(file_extension):
                continue
            os.remove(os.path.join(directory_path, file))

        os.rmdir(directory_path)
