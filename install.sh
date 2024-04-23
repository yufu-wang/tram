# conda remove -n tram --all -y
# conda create -n tram python=3.10 -y
# conda activate tram

conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

conda install pytorch-scatter -c pyg
conda install -c conda-forge suitesparse

pip install pulp
pip install supervision

pip install open3d
pip install opencv-python
pip install loguru
pip install chumpy
pip install einops
pip install plyfile
pip install pyrender
pip install segment_anything
pip install scikit-image
pip install smplx
pip install timm==0.6.7
pip install evo
pip install pytorch-minimize
pip install imageio[ffmpeg]
pip install numpy==1.23
pip install gdown

