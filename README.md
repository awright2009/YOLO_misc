# YOLO_misc

Misc code

	DepthToCloud.py -- Takes 16 bit depthmap image and corresponding rgb image and displays the depth and image as a point cloud

	DepthToNormal.py -- Takes 16 bit depthmap image and generates a mesh colored by normal color and writes the image to disk.
		See images/left_small_normal.png and images/right_small_normal.png

	ObjToDepth.py -- Takes a OBJ file and writes the depth map to disk (used with mesh/depth_scene.obj to generate images/16depth.png)

	depth_min_max.py -- Takes 16 bit depth image that has been segmented by YOLOv11 or SAM2 to get the min/max depth values for the masked object

	get_depth.sh -- Generates command line for above for each isolate*png

	normal_segmentation.py -- Takes the normal image and groups normals together based on similar directionality, idea being it can form as a method of segmenting large planes (not great)
		see images/left_segmented_labels.png images/right_segmented_labels.png

	YOLOv11 directory contains ultralytics YOLOv11 python scripts for image segmentation and isolation

	YOLOv3 directory contains python scripts for running YOLOv3 and environment setup

	sam2 directory contains ultralytics sam2 python scripts for image segmentation and isolation

	mesh - contains wavefront obj meshes (some 7ziped depth meshes due to size over 100mb)

	DepthAnythingV2 - Has scripts for running DepthAnythingV2 that outputs 16 bit depth images for a given input image

	images - has various images generated from left.JPG and right.JPG, isolated and object detections, as well as normal and depth maps from DepthAnythingV2

	images/sam2/ - has isolated images output from sam2