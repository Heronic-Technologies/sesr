#Install tensorflow-datasets first:
pip install tensorflow_datasets
pip install packaging
pip install onnx
pip install tf2onnx

#Now install tensorflow-gpu with conda (please install version 2.3 or above)
#Conda is used here since it takes care of correct cudatoolkit and cudnn
pip install tensorflow[and-cuda]==2.14.0
pip install numpy==1.26.4

