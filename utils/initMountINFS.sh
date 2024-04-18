mkdir data
mkdir syn_data
cp /opt/data/private/imagenet.tar.gz data/
# already divided into 1000 classes folders
cd data
tar -xzf *gz
rm *gz
cd ..
