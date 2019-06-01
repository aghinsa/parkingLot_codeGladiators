# YOLOv3 Readme  

### Installing Darknet   

~~~~
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make
~~~~ 

* To activate GPU usage
    *  set *GPU = 1* in *MakeFile* 
* To change checkpoint interval  
    * edit line 136 in detector.c  

> run ./darknet to test installation  
expected output *usage: ./darknet < function >*  

### Data Format  
* Each image should have a *.txt* file with the same name,containing  
    * > object_class,centre_x,centre_y,width,height  
    * object_class : index from 0
    * all other should be normalized to 0-1 range.
    * [centre_x,width]/=image_width
    * [centre_y,height]/=image_height 

### Configaration  

#### yolo_v3.cfg  
* make the following changes  
   ~~~~
    Line 3: set batch=_batch_size
    Line 4: set subdivisions=_sub_division, the batch will be divided by subdivisions
    Line 603: set filters=(classes + 5)*3 
    Line 610: set classes=_classes 
    Line 689: set filters=(classes + 5)*3 
    Line 696: set classes=_classes
    Line 776: set filters=(classes + 5)*3 
    Line 783: set classes=_classes 
  ~~~~  

  > create parkLot.data  
  ~~~~
    classes= 2 
    train  = /media/aghinsa/DATA/workspace/ParkingLot/parkLot_yolov3/parkLot_train.txt  
    valid  = /media/aghinsa/DATA/workspace/ParkingLot/parkLot_yolov3/parkLot_test.txt  
    names = /media/aghinsa/DATA/workspace/ParkingLot/parkLot_yolov3/parkLot.names  
    backup = /media/aghinsa/DATA/workspace/ParkingLot/parkLot_yolov3/final_weights/
  ~~~~

  > parkLot.names  (Names of classes)
  ~~~~
    not_occupied
    occupied
  ~~~~  
  > train,test .txt contains absolute paths of images (no need for .txt paths) seperated by newline  

  > run
  ~~~~
    ./darknet detector train ../parkLot.data ../parkLot_yolov3.cfg ../darknetWeights/parkLot_yolov3_6400.weights
  ~~~~
  > To change the anchors run,and copy paste results to the cfg file
  ~~~~
  ./darknet detector calc_anchors ../parkLor.data -num_of_clusters 9 -width 608 -height 608 -show
  ~~~~

## Detection  
* Change *tunable params* in *detection.py*  
* to run model on a single image  
> python detection.py --image=/path_to_image  
* for batch  
> python detection.py --images_dir=path --save_dir=path