# Image Locate
Cross-view Image Geolocation is the identification of the location of a ground based image by finding a matching aerial 
photo.

### GPU Setup
This project runs within a docker container provided by Tensorflow and requires the use of a GPU.  Before running, 
ensure that you have a GPU compatible with Tensorflow and that is has been set up as per the instructions:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html provided by NVIDIA.

### Data Setup
The data used in this study was from the CVUSA dataset from Workman et al. in their paper "Wide-Area Image 
Geolocalization with Aerial Reference Imagery" (http://arxiv.org/abs/1510.03743).  Access to the data can be requested
here: https://mvrl.cse.wustl.edu/datasets/cvusa/ 

To build and run the container, first modify the following line of the `docker-compose.yml`:
```yaml
volumes:
  - "./:/tf/notebooks"
  - "/media/your/data/file/:/tf/CVUSA/"    # <== Place your file path to your data here
```    
Your data file should be laid out as follows:
```
CVUSA/
├── clean_aerial/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── image_3.jpg
└── clean_ground/
    ├── image_1.jpg
    ├── image_2.jpg
    └── image_3.jpg
```
Note that each ground and aerial image have the same name, this allows for the loading of the dataset in proper order. 
You can include sub-folders at your own risk, however this has caused problems with some of the dataset not loading 
correctly in the past.

### Running the code
To build and run the project, find the `docker-compose.yml` file and run:
```bash
docker-compose up --build
```
This will build and start a container named `picture-locate` and will provide access to the jupyter server though a link.
Training can be run by entering the container and running the `trainModel.py` file as follows:

```bash
docker exec -it image-locate bash
```
This will enter the container and provide a prompt at the `notebooks/` directory.dd
```bash
________                               _______________
___  __/__________________________________  ____/__  /________      __
__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /
_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ /
/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/


WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.

To avoid this, run the container by specifying your user's userid:

$ docker run -u $(id -u):$(id -g) args...

root@1ec13bf1ef31:/tf/notebooks# 
```
