# Smartbin
Smartbin AI is a personal project aiming at training an artifial intelligence from scratch.
It relies on both Tensorflow and OpenCV for Python 3.

The following document covers every step from gathering images to training the AI and testing it.

## Installation
The following instructions have been tested on Ubuntu 18.04.

Clone this repository.

```bash
sudo apt -y install git
git clone https://github.com/Cybermated/Smartbin_AI.git
```

Make sure Python 3 is installed on your machine as well as Pip for Python 3.
```bash
sudo apt -y install python3-minimal
python3 --version
Python 3.x.x
```

Setup Pipenv and restore the environment from requirements.txt. Install errors may raise during the process.
```bash
sudo apt install -y python3-pip
pip3 install --user pipenv
cd Smartbin_AI
pipenv install -r requirements.txt --skip-lock --clear
pipenv shell
```

Install protobuf. It is required for the training.
```bash
sudo apt install -y curl
cd /tmp
# Make sure you grab the latest version.
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protoc-3.7.1-linux-x86_64.zip
unzip protoc-3.7.1-linux-x86_64.zip -d protoc3
sudo mv protoc3/bin/* /usr/local/bin/
sudo mv protoc3/include/* /usr/local/include/

# Optional: change owner.
sudo chown $USER /usr/local/bin/protoc
sudo chown -R $USER /usr/local/include/google

# Check version.
protoc --version
libprotoc 3.7.1
```

Install Tensorflow models.
```bash
mkdir tensorflow
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
```

Install the pycocotools. It is in the requirements file but it may have failed.
```bash
cd /tmp
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools tensorflow/models/research/
```

Export to PYTHONPATH tensorflow/models/research/slim:
```bash
cd tensorflow/models/research
python3 setup.py install
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Edit the config.py by filling the the array with the classes you want to detect (i.e. dog, cat, sheep...).
```python
ROI_CLASSES = []
```

Run generate_labelmap.py:
```bash
python3 generate_labelmap.py
```

A file named smartbin-labelmap.pbtxt must have appeared in /training/config. It links

To be continued.

## Usage
#### Collect images

To be added.

#### Generate folders

To be added.

#### Export ROIs

To be added.

### Train model

To be added.

### Test model

To be added.

## Contributing

To be added.

## License

To be added.
