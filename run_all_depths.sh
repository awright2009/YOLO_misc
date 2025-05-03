#!/bin/bash

# Set base paths
RGB_DIR="./COLMAP/south_building"
DEPTH_DIR="./COLMAP/south_building_depth"
METRIC_DEPTH_DIR="./COLMAP/south_building_metric_depth"

# Loop through all JPG RGB images
for rgb_file in "$RGB_DIR"/*.JPG; do
    rgb_filename=$(basename "$rgb_file")
    base_name="${rgb_filename%.*}"  # Remove extension

    depth_file="$DEPTH_DIR/${base_name}.png"
    metric_depth_file="$METRIC_DEPTH_DIR/${base_name}.png"

    # Run with normal depth file
    if [[ -f "$depth_file" ]]; then
        echo "Processing standard depth: $base_name"
        python DepthToPlyBin.py "$rgb_file" "$depth_file" 0 0

        ply_file="${base_name}.ply"
        if [[ -f "$ply_file" ]]; then
            mv "$ply_file" "$DEPTH_DIR/$ply_file"
            echo "Moved $ply_file to $DEPTH_DIR"
        fi
    else
        echo "Warning: Depth file missing for $base_name"
    fi

    # Run with metric depth file
    if [[ -f "$metric_depth_file" ]]; then
        echo "Processing metric depth: $base_name"
        python DepthToPlyBin.py "$rgb_file" "$metric_depth_file" 0 0

        ply_file="${base_name}.ply"
        if [[ -f "$ply_file" ]]; then
            mv "$ply_file" "$METRIC_DEPTH_DIR/$ply_file"
            echo "Moved $ply_file to $METRIC_DEPTH_DIR"
        fi
    else
        echo "Warning: Metric depth file missing for $base_name"
    fi
done
