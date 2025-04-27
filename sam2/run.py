from ultralytics import SAM

#sam = SAM("sam2_b.pt")
sam = SAM("sam2.1_l.pt")

sam.info()

#model(source=0, show=True, save=True)


results = sam.predict(source="left.JPG", show=False, save=True)

for r in results:
    print(f"Detected {len(r.masks)} masks")

print(results)

for result in results:
    print("result: ", result)
    print("names: ", result.names);
    for box in result.boxes:
        print("class  has box ", box.xyxy.tolist())
    for mask in result.masks:
        print("class  has mask ", mask)

