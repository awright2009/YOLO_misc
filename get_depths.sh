for file in ./images/isolate3_*depth.png; do 
    if [ -f "$file" ]; then 
	echo "python DepthToAABB.py \"$file\""
        #python DepthToAABB.py "\"$file\""
    fi 
done


