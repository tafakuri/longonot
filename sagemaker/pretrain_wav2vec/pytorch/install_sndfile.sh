%%bash
wget 'https://github.com/libsndfile/libsndfile/releases/download/1.0.31/libsndfile-1.0.31.tar.bz2'
tar -xf libsndfile-1.0.31.tar.bz2
cd libsndfile-1.0.31/
./configure
make
sudo make install