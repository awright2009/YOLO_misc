# YOLO_misc

# Intro

So, needed a project for a class and we were learning about YOLO and I thought to myself I wanted to get YOLO running and stick with CNN and image processing/computer vision. Me being me, I thought we should make the YOLO detections 3D, from there I started looking at Depth cameras and stumbled into DepthAnythingV2, which sounds awesome and makes this project not require a depth camera. The object detections from YOLO aren't great (they really need to be pixel perfect in order to not pull depth from something else), but making a 3d bounding box from a point cloud is rather easy as it's just the min/max of the X, Y, Z from the depth mask. Note that SegmentAnything2 from Meta does a much better job with the masking of objects, but I used the ultralytics version which just gave numeric object id's that I didn't find super useful. So I stuck with YOLO and just scaled down the detection mask by 80%, note that this doesn't work great with objects like bicycles as they are not solid objects. But works great for solid things like cups.

Anyway. Eventually, I stumbled into Nvidia's NeRF, which seems to use something called COLMAP to do the heavy lifting (determining camera position and orientation from a series of photogrammetry images) So I started messing with COLMAP, which has a Windows program that after 5 hours or so of CUDA enabled GPU work it will create a point cloud and solve the camera positions using triangulation and SiFT. So I think I can use those camera poses from their text files and do the same sort of thing with my point clouds. However, you may want to note that using COLMAP to generate an image is not real-time. So for objects that move, using a single RGB image or stereo RGB images will be much much faster. But if you can generate a map offline, COLMAP, NeRF, and Gaussian Splatting are the way to go.

But, just yesterday I found out about Guassian splatting, which seems like the right way to do point cloud rendering, splatting has been used previously for rendering fluids and clouds, but this takes your 3d point cloud and makes the points Gaussians which can then be corrected using gradient descent. Note that they really use ellipsoids which they scale and rotate as Gaussians can become invalid due to not being invertible. Rendering the Gaussians directly is fairly fast as well, the biggest gating item is the memory requirements for storing all the data, which graphics cards can handle pretty readily now as that corresponds with the requirements for training Neural Networks. I'll have to read the paper and see if I can convert my point clouds to Guassians and render them like the paper does, but for the most part, my original goal of getting boxes on YOLO detections works pretty well assuming you have a good mask. At least for me taking a point cloud, and converting them to ellipsoids of a unit length seems easy enough, but using backpropagation to reshape them doesn't jump out to me as something easy to do.

# Videos of Point Cloud Rendering
https://youtu.be/gnjmazjuQhY

# Python Code

	CloudViewer.py -- Takes a binary PLY file and renders it similar to DepthToCloud.py

	DR_StereoPointCloud_Match_Target.py -- (non functional currently) Working code for differentiable rendering (ie: match point cloud to target using gradient descent)

	DepthToAABB.py -- Takes masked depth images and converts them to AABB's stored in object_depths.txt for loading by Point Cloud viewers

	DepthToCloud.py -- Takes 16 bit depthmap image and corresponding rgb image and displays the depth and image as a point cloud
	
	DepthToCloudImage.py -- Same as above, but just renders to a output image instead of viewing live
	
	DepthToCloudStereo.py -- Same as DepthToCloud, but loads two RGB and depth images to allow for matching between the sets

	DepthToNormal.py -- Takes 16 bit depthmap image and generates a mesh colored by normal color and writes the image to disk.
		See images/left_small_normal.png and images/right_small_normal.png

	DepthToPly.py -- Takes 16 bit depthmap and RGB image and writes it out to a binary PLY file

	NormalToSegment.py -- Takes a Normal image and groups the normals into groups based on angle tolerance value writes out the groups to a file
	
	SegmentToMask.py -- Takes the grouped normals and generates isolated masks of the top four by pixel count
	
	ObjToDepth.py -- Takes a OBJ file and writes the depth map to disk (used with mesh/depth_scene.obj to generate images/16depth.png)

	cpu_melter.py -- Test code for differentiable rendering, takes a few randomly initialized triangles and attempts to match the target image.
		(see cpu_melter.png which was run against images/left.JPG) -- Note runs on CPU and takes a long time

	get_depths.sh -- This just prints DepthToAABB.py commands for each images/isolate3_*depth.png

	run_all.sh -- This is intended to run all operations, could connect with webcam2 and attempt to get real time output

	run_all_depths.sh -- This is just for convience of converting all the images in the COLMAP directory

	stereo.py -- This is opencv traditional stereo image depth generation (results not good)
	
	test_cuda.py -- This is just a check to be sure CUDA is available (specifically for DR_StereoPointCloud_Match_target.py)
	
	webcam2.py -- This just runs bounding box yolo11 against a webcam (Figure my Mac Mini M4 is a good YOLO webcam box) -- but could also be put together with run_all.sh to run the whole shebang
	
	yolo3d.pdf -- pdf slides, don't have the differentiable rendering addendum though which was last minute

# Directories
	COLMAP - Colmap sample images, pose extractor (from images.txt) and point clouds (ply files)
	
	DepthAnythingV2 - Scripts used to run DepthAnythingV2
	
	RANSAC -- RANSAC test code (just divides point cloud by a plane)
	
	VideoToImages -- ffmpeg bat and sh file to take a video and get frame images every 500ms (also has ffmpeg.exe in zip)
	
	YOLOv11 -- Scripts for running YOLOv11, isolate3.py gets detection masks scaled down 80%
	
	YOLOv3 -- Scripts / environment setup for running darknet / YOLOv3
	
	images -- lots of images, notably left.JPG and right.JPG, gopro images have capitalized JPG suffix. left_metric.png / right_metric.jpg and left.png right.png are depth images from depth anything v2 (metric and non metric)

	kinect -- original kinect camera frame extraction code (DepthWithColor-D3D-bin.zip) Python scripts convert raw file to images (ViewRGBX.py / ViewDepth.py) data has some saved frames

	left_groups -- plane masks generated from normals (top four groups by pixel count) from left image
	
	right_groups -- plane masks generated from normals (top four groups by pixel count) from left image

	matlab -- matlab point cloud viewer code
	
	mesh - obj files generated during normal creation (left.7z / right.7z) depth_scene.obj is the cubes / cylinder example image from powerpoint / proposal
	
	papers - Papers I thought were related / worth reading. 3D Gaussian Splatting, NeRFs, Screen space fluid rendering (essentially splatting), Screen space meshes (essentially precursor to splatting)
	
	rendered_output - output directory for DR_StereoPointCloud_Match_target.py -- which is my differentiable rendering test code (needs work)
	
	sam2 -- scripts I used for object masks from SAM2 using ultralytics API's

# Gaussian Splatting Link

https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

# Good Inverse Rendering / Differentiable Rendering run down

https://jjbannister.github.io/tinydiffrast/
