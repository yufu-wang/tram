#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL Neutral model
echo -e "\nYou need to register at https://smplify.is.tue.mpg.de"
read -p "Username (SMPLify):" username
read -p "Password (SMPLify):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O './data/smpl/smplify.zip' --no-check-certificate --continue
unzip data/smpl/smplify.zip -d data/smpl/smplify
mv data/smpl/smplify/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl data/smpl/SMPL_NEUTRAL.pkl
rm -rf data/smpl/smplify
rm -rf data/smpl/smplify.zip

# SMPL Male and Female model
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O './data/smpl/smpl.zip' --no-check-certificate --continue
unzip data/smpl/smpl.zip -d data/smpl/smpl
mv data/smpl/smpl/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl data/smpl/SMPL_FEMALE.pkl
mv data/smpl/smpl/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl data/smpl/SMPL_MALE.pkl
rm -rf data/smpl/smpl
rm -rf data/smpl/smpl.zip

# Thirdparty checkpoints
wget -P ./data/pretrain/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
wget -P ./data/pretrain/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
gdown --fuzzy -O ./data/pretrain/droid.pth https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing
gdown --fuzzy -O ./data/pretrain/camcalib_sa_biased_l2.ckpt https://drive.google.com/file/d/1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO/view?usp=sharing

# Our checkpoint and an example video
gdown --fuzzy -O ./data/pretrain/vimo_checkpoint.pth.tar https://drive.google.com/file/d/1fdeUxn_hK4ERGFwuksFpV_-_PHZJuoiW/view?usp=share_link
gdown --fuzzy -O ./example_video.mov https://drive.google.com/file/d/1H6gyykajrk2JsBBxBIdt9Z49oKgYAuYJ/view?usp=share_link