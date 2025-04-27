
colorImage = imread("right_color.jpg");
imshow(colorImage)

imfinfo("right_depth.png")
depthImage = imread("right_depth.png");

depthImage = rgb2gray(depthImage);

imshow(depthImage)


imageSize = [3000, 4000]; % [height, width]
sensorSize = [4.55, 6.17]; % mm [height, width]
focalLengthMM = 1.5;

fx = (focalLengthMM / sensorSize(2)) * imageSize(2);
fy = (focalLengthMM / sensorSize(1)) * imageSize(1);
principalPoint = [imageSize(2)/2, imageSize(1)/2]; % [cx, cy]
focalLength = [fx, fy];

intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);



depthScaleFactor = 10.0 / 1.0
maxCameraDepth   = 5e5


ptCloud = pcfromdepth(depthImage,depthScaleFactor, intrinsics, ...
                      ColorImage=colorImage, ...
                      DepthRange=[0 maxCameraDepth])



pcshow(ptCloud, VerticalAxis="Y", VerticalAxisDir="Up", ViewPlane="YX")

viewer = pcviewer(ptCloud)