for file in ./images/*depth.png; do 
    if [ -f "$file" ]; then 
	echo "python depth_min_max.py $file"
        #python depth_min_max.py "$file"
    fi 
done


