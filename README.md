# TFG

!git clone https://github.com/marGaliana/SignLanguageProcessing.git
%cd SignLanguageProcessing/
!git checkout feature/11_XGBoost
%cd SignGestureDetection/
%mkdir Assets/
%cd Assets
%mkdir Dataset/
%mkdir NeuralNetworkModel/
%cd Dataset
%mkdir Pickels/
!git clone https://github.com/marGaliana/SignLanguageProcessingDataset.git
!mv SignLanguageProcessingDataset Gesture_image_data
%cd ../../
