wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 -s
rm Miniconda3-latest-Linux-x86_64.sh
conda=miniconda3/bin/conda
$conda create -y -n med python=3.11.9 pip
$conda install -n med -y pytorch tensorflow=2.16.1 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

source miniconda3/bin/activate med

pip install medmnist ACSConv
pip install timm
