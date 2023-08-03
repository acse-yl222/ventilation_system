# ventilation_systerm

## Introduction


## Installation
This project is based on Pytorch.
You can use the following command to install the Pytorch.

### create a virtual enviroment
```bash
conda create -n ai4pdes python=3.9
conda activate ai4pdes
```

windows user
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

linux user
```bash
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

MacOS user
```bash
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

## Usage
first download the weight file.
```bash
python download_weight.py
```
or you can download through this google drive link and move it to the weight folder.
[google drive link](https://drive.google.com/file/d/1eW55eq7pHaBEba99B7svK_tAL9yRy36q/view?usp=sharing)

the conrect the weight path in [this file](main.py)

And then you can run the main.py to see the result.
```bash
python main.py
```