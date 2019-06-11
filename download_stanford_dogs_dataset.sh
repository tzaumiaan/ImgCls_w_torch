mkdir -p data
mkdir -p data/sdd
cd data/sdd
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
tar xvf images.tar
tar xvf lists.tar
cd ../..
