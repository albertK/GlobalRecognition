# GlobalRecognition

## 1. Installation
Before installing this package, make sure you have PCL-1.8 installed. 

	git clone https://github.com/PointCloudLibrary/pcl pcl-trunk
	cd pcl-trunk && mkdir build && cd build
	cmake -DCMAKE_BUILD_TYPE=Release .. 
	make
	sudo make install

To install this package, type

	git clone https://github.com/albertK/GlobalRecognition.git
	mkdir build
	cd build
	cmake ..
	make

##2. Usage

###2.1 Convert CAD model(.ply) to point cloud
To convert your .ply file to pcd, first put your .ply file under /models

	./train_db

