for file in ./images/*depth.png; do 
    if [ -f "$file" ]; then 
        python depth_min_max.py "$file"
    fi 
done


