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

Para la estrategia AccuracyDecisionTree hace falta instalar graphviz para poder mostrar (plot) el modelo de abol de decisión. En el caso de un mac se tiene que hacer con: brew install graphviz

conda install graphviz python-graphviz