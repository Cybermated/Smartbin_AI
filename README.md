# Smartbin

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/30c5c586a0b94f2db6801bf31512dd3d)](https://www.codacy.com/app/Cybermated/Smartbin_AI?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Cybermated/Smartbin_AI&amp;utm_campaign=Badge_Grade)

Smartbin AI is a personal project aiming at training an artifial intelligence from scratch to detect 
garbage (plastic bottles, cans, tin cans, plastic containers...).
It relies on both Tensorflow and OpenCV for Python 3.

The following document covers every step from gathering images to training the AI and testing it.

## Installation
The instructions below have been tested on CentOS 7 and Ubuntu 18.04 (with and without graphical interface).

Start by cloning this repository.

```bash
sudo apt install -y git-core
git clone https://github.com/Cybermated/Smartbin_AI.git
```

Make sure Python 3 is installed on your machine as well as Pip for Python 3.
```bash
sudo apt install -y python3-minimal python3-setuptools python3-dev
python3 --version
Python 3.x.x
```

On headless servers make sure to install the following packages.
```bash
# For Debian-based installations.
sudo apt install -y libsm6 libxrender1 libfontconfig1
# For Redhat-based installations.
sudo yum install -y libXext libSM libXrender
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
sudo apt install -y autoconf automake libtool curl make g++ unzip
cd /tmp
# Make sure you grab the latest version.
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protoc-3.7.1-linux-x86_64.zip
unzip protoc-3.7.1-linux-x86_64.zip -d protoc3
sudo mv protoc3/bin/* /usr/local/bin/
sudo mv protoc3/include/* /usr/local/include/

# Optional: change owner.
sudo chown $USER /usr/local/bin/protoc
sudo chown -R $USER /usr/local/include/google

# Refresh librairies.
sudo ldconfig

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

__The following commands must be executed in the Pipenv you created earlier.__

Install the pycocotools. It is in the requirements file but it may have failed.
```bash
sudo apt install -y gcc
cd /tmp
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools path/to/tensorflow/models/research/
```

Export to PYTHONPATH Tensorflow models:
```bash
cd path/to/tensorflow/models/research
python3 setup.py install
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
source ~/.bashrc
```


## Usage
#### Collect images

The first step is to gather images of what you want to detect. In this "tutorial" we will take for examples dogs, cats 
and sheeps. Start by editing the config.py file and change the following line.

```python
ROI_CLASSES = ['sheep', 'dog', 'cat']
```

Run generate_labelmap.py:
```bash
python3 generate_labelmap.py
```

A file named "labelmap.pbtxt" must have appeared in the training/config folder.
It is possible to add new classes afterwards but the labelmap file must be regenerated as it maps
each class name to an internal ID for Tensorflow. Also, make sure to not modify the order of the classes.

Then, you must gather images of sheeps, cats and dogs. Images must be big enough and you should have several kinds of 
sheeps, cats and dogs. We recommend at least 2000 images per class but you can start with 100 if you plan to use the 
augmentation script.

Put the images in the pretraining/raw_images folder.


You can also shoot videos by yourself, in this case put them in 
the pretraining/raw_videos folder. Then run the extract_frames.py.
```bash
python3 extract_frames.py
```

New images must have appeared in the raw_images folder.

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
