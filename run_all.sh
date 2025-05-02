#!/bin/bash

# Paths (avoid spaces around "=" and use absolute paths without ~)
depthanything_dir="$HOME/Downloads/Depth-Anything-V2-main"
depthanything_metric_dir="$HOME/Downloads/Depth-Anything-V2-main/metric_depth"
yolo_dir="$HOME/yolo11"
work_dir="$HOME/yolo"


# assume we get some left / right images from a camera to this path
left_image="$HOME/yolo/images/left.JPG"
right_image="$HOME/yolo/images/right.JPG"

# Copy images to the appropriate directories
cp "$left_image" "$depthanything_metric_dir/input/"
cp "$right_image" "$depthanything_metric_dir/input/"

cp "$left_image" "$yolo_dir/"
cp "$right_image" "$yolo_dir/"

# Change to DepthAnything directory
cd "$depthanything_dir" || { echo "Failed to cd into $depthanything_dir"; exit 1; }

# Set up and activate virtual environment
python3 -m venv venv
source venv/bin/activate


cd "$depthanything_metric_dir" || { echo "Failed to cd into $depthanything_metric_dir"; exit 1; }

# Run depth estimation metric version (metric directory)
python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth  --max-depth 20   --img-path ./input/ --outdir ./output/  --pred-only --grayscale

# Copy output images to YOLO directory
cp ./output/left.png "$yolo_dir/left_metric.png"
cp ./output/right.png "$yolo_dir/right_metric.png"

# Change to DepthAnything directory
cd "$depthanything_dir" || { echo "Failed to cd into $depthanything_dir"; exit 1; }

# Deactivate virtual environment
deactivate

# Change to YOLO directory
cd "$yolo_dir" || { echo "Failed to cd into $yolo_dir"; exit 1; }

# Run isolate3 on both images
python isolate3.py "$left_image"
python isolate3.py "$right_image"

# Move generated isolate images
cp isolate3*png "$work_dir/images/"

left_depth_image="${left_image%.*}_depth.png"
right_depth_image="${left_image%.*}_depth.png"

#Change to work directory
cd "$work_dir" || { echo "Failed to cd into $work_dir"; exit 1; }

# This will generate a new object_depth.txt for each detected object
rm object_depths.txt

for file in ./images/isolate3_*depth.png; do 
    if [ -f "$file" ]; then 
	echo "python DepthToAABB.py "\"$file\"""
        python DepthToAABB.py "$file"
    fi 
done


# These take a long time
#python DepthToNormal.py "$left_depth_image" 5
#python DepthToNormal.py "$right_depth_image" 5

#python NormalToSegment.py "images/left_small_normal.png"
#python NormalToSegment.py "images/right_small_normal.png"

#python SegmentToMask.py "images/left_small_normal_segmented_labels.png"
#python SegmentToMask.py "images/right_small_normal_segmented_labels.png"


# Render point cloud image with custom perspective
python DepthToCloudImage.py "$left_image" "$left_depth_image" 0 1 0 0 -50
python DepthToCloudImage.py "$right_image" "$right_depth_image" 0 1 0 0 -50
