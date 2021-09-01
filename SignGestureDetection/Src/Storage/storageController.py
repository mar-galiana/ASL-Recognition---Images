import os
import zipfile
from Storage.storageEnum import FileEnum


class StorageController:

    def __init__(self):
        pass

    @staticmethod
    def compress_files(files, compressed_file_path):

        # create a ZipFile object
        zip_obj = zipfile.ZipFile(compressed_file_path, 'w')

        for file in files:
            file_path = file[FileEnum.FILE_PATH.value]
            file_name = file[FileEnum.FILE_NAME.value]
            zip_obj.write(file_path + file_name, file_name, compress_type=zipfile.ZIP_DEFLATED)

        # close the Zip File
        zip_obj.close()

    @staticmethod
    def remove_files_from_folder(files):
        for file in files:
            file_path = file[FileEnum.FILE_PATH.value]
            file_name = file[FileEnum.FILE_NAME.value]
            os.remove(file_path + file_name)
