## HGC-Net: Deep anthropomorphic hand grasping in clutter

### Installation

Install necessary packages: 
```sh
pip install -r requirements.txt
```

Install pointnet++:
```sh
cd pointnet2/pointnet2/
pip install -e .
```

### Data processing
We provide dataset for hand grasps and cluttered scenes at https://drive.google.com/file/d/1tqM67rb4XRyiWu640QTNcAOrwCBF4mc9/view?usp=sharing
Please download the data and place it in the same folder with this repo.

You can run 
```sh
vis_object_grasp.py
```
to visualize the grasp data for each object.

#### Collison Detection

Run 
```sh
collision_detection.py
```
to transform grasps to the scene and filter collision grasps. It will generate a file 'scene_grasps' that contains collision free grasps for each scene. It might take several hours for collision checking. Once finished, you can run 
```sh
vis_scene_grasps.py
```
to visualize the scene grasps.

#### Match Point Grasp
Run 
```sh
match_point_grasp.py
```
to generate point labels for each point cloud (for each scene, we capture 4 images from different views and convert them to point cloud). These process can be done online but it takes time during trainning, so we do it offline. It will generate a file 'point_grasp_data' contains grasp labels.

### Training

Just run 
```sh
train.py & vis.py 
```

to train the model and visualize it. Note that I did not test this part but it should be work. 


