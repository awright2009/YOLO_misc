# YOLO_misc

# Intro

So, needed a project for a class and we were learning about YOLO and I thought to myself I wanted to get YOLO running and stick with CNN and image processing / computer vision. Me being me, I thought we should make the YOLO detections 3D, from there I started looking at Depth cameras and stumbled into DepthAnythingV2, which sounds awesome and makes this project not require a depth camera. The object detections from YOLO aren't great (really need to be pixel perfect in order to not pull depth from something else), but making a 3d bounding box from a point cloud is rather easy as it's just the min/max of the X,Y,Z from the depth mask. Note that SegmentAnything2 from Meta does a much better job with the masking of objects, but I used the ultralytics version which just gave numeric object id's that I didn't find super useful. So I stuck with YOLO and just scaled down the detection mask by 80%, not that this doesn't work great with objects like bicycles as they are not solid objects. But works great for solid things like cups.

Anyway. eventually I stumbled into Nvidia's NeRF, which seems to use something called COLMAP to do the heavy lifting (determining camera position and orientation from a series of photogrammetry images) So I started messing with COLMAP, which has a windows program that after 5 hours or so of CUDA enabled GPU work it will create a point cloud and solve the camera positions using triangulation and SiFT. So I think I can use those camera poses from their text files and do the same sort of thing with my point clouds. However, you may want to note that using COLMAP to generate an image is not real time. So for objects that move, using a single RGB image or stereo RGB images will be much much faster. But if you can generate a map offline, COLMAP, NeRF and Gaussian Splatting is the way to go.

But, just yesterday I found out about Guassian splatting, which seems like a good idea, splatting has been used previously for rendering fluids and clouds, but this takes your 3d point cloud and makes the points gaussians which can then be corrected using gradient descent. Note that they really use ellipsoids which they scale and rotate as Gaussians can become invalid due to not being invertible. Rendering the guassians directly is fairly fast as well, the biggest gating item is the memory requirements for storing all the data, which graphics cards can handle pretty readily now as that corresponds with the requirements for training Neural Networks. I'll have to read the paper and see if I can convert my point clouds to guassians and render them like the paper does, but for the most part my original goal of getting boxes on YOLO detections works pretty well assuming you have a good mask.


# Misc Code
(need to update this again as things have progressed since then)

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



https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

