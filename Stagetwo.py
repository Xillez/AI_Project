from PIL import Image, ImageChops
import math
import numpy as np
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import time
import matplotlib.pyplot as plt
import cv2
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

images = dict()
skeletonizeImage = dict()

def Skeletonize(img):
	tempImg = cv2.imread(img, 0)

	h, w = tempImg.shape[:2]
	if w == 0 or h == 0 :
		e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
	else:
		e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))

	 # estimate human poses from a single image !
	image = common.read_imgfile(img, None, None)
	if image is None:
		print('[Stage --][Errors] Image can not be read, path=%s' % img)
		sys.exit(-1)
	t = time.time()

	print('[Stage --][Info] Getting intrest points for each human in picture.')
	humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
	elapsed = time.time() - t

	#blank_image = np.zeros((h,w,3), np.uint8)


	#image = TfPoseEstimator.draw_humans(blank_image, humans, imgcopy=False)

	print('[Stage --][Info] Filtering intrest points from the picture.')
	intrestPoint = []
	for human in humans:
		for i in range(common.CocoPart.Background.value):
			if i not in human.body_parts.keys():
				continue

			print(i)
			bodyPart = human.body_parts[i]
			intrestPoint.append((i, (int(bodyPart.x * w + 0.5))))
			intrestPoint.append((i, (int(bodyPart.y * h + 0.5))))

	if len(skeletonizeImage) == 0 :
		skeletonizeImage[0] = intrestPoint
	else  :
		skeletonizeImage[1] = intrestPoint

	print(intrestPoint)
	print(len(skeletonizeImage))
	#print(len(intrestPoint))
	#print(len(image.flatten()))
	#print(len(np.trim_zeros(image.flatten())))
	##cv2.imwrite("Temp.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def DifferenceBetweenTwoFrames():#(key, values):

#	if key not in images :		## if dont have the image of type key / if there is no current frame.
#		images.append((key, value))					## add the frame as current frame.
#		return False
#	else :									## Other wise
#		img1 = images[key]					## remove the previous frame.
#		img2 = values						## get the current frame.
#		del images[key]						## delete the previous frame from store.
#		images.append((key, value))			## save the current frame as previous for next call.
	
	
	## 
	## TODO : Skeletonize the fetched pictures.
	##


	Skeletonize('./img0059.png')
	Skeletonize('./img0060.png')
	##ValidateSkeletonzie();

	##print(len(img1.flatten()))
	#print(len(newimg1.flatten()))
	
	##tempimg1 =	[0.]
	##tempimg2 =	[0.]
	print(skeletonizeImage[0])
	img1 = np.insert(0., 1, skeletonizeImage[0])
	img2 = np.insert(0., 1, skeletonizeImage[1])

	##print(img1)

	##print(img1)
	distance, paths = dtw.warping_paths(img1, img2)
	##print(distance)
	print(paths)
	#dtwvis.plot_warping(img1, img2, path, filename="warp.png")


## make both skeletons of same length.
def ValidateSkeletonzie():
	tempList = []

	if len(skeletonizeImage[0]) > len(skeletonizeImage[1]) :
		tempList = skeletonizeImage[0]
	else :
		tempList = skeletonizeImage[1]


	#print(index)
	#print(largestList)

	#tempList = [(i, v) for counter, i, v in enumerate(range(skeletonizeImage[0])) if i not in skeletonizeImage[1][counter]]	
	
	for k, v in tempList:
		if k not in skeletonizeImage[0]:
			tempList.append(skeletonizeImage[x])
	print(tempList)

DifferenceBetweenTwoFrames()

#def DifferenceBetweenTwoFrames(key, values):
#	global images
#	##imgaes.append(img);
#
#	if(key not in images) : 
#		images.append((key, values))
#		return False;
#	else :
#		img1 = images.pop(key)
#		img2 = images.pop()
#
#
#	#img1 = Image.open('./testvideoimages/img0001.png')
#	#img2 = Image.open('./testvideoimages/img0002.png')
#	#print(img1)
#
#
#	img = ImageChops.difference(img1,img2)
#	img.save('test_diff2.png') 
#	print(image_entropy(img)) # 5.76452986917

















'''




	W = 20.
	img1 = cv2.imread('./testvideoimages/img0001.png')
	img2 = cv2.imread('./testvideoimages/img0002.png')

	height, width, depth = img1.shape
	imgScale = W/width
	newX,newY = img1.shape[1]*imgScale, img1.shape[0]*imgScale
	newimg1 = cv2.resize(img1,(int(newX),int(newY)))
	newimg2 = cv2.resize(img2,(int(newX),int(newY)))

	
	cv2.imwrite('com1.png', newimg1,  [cv2.IMWRITE_PNG_COMPRESSION, 9])
	cv2.imwrite('com2.png', newimg2,  [cv2.IMWRITE_PNG_COMPRESSION, 9])

	##print(len(img1.flatten()))
	print(len(newimg1.flatten()))
	
	##tempimg1 =	[0.]
	##tempimg2 =	[0.]

	img1 = newimg1.flatten()
	img2 = newimg2.flatten()


	img1 = np.insert(0., 1, img1)
	img2 = np.insert(0., 1, img2)

	##print(img1)

	##print(img1)
	##path = dtw.warping_path(img2, img2)
	##print(path)
	##distance, paths = dtw.warping_paths(img1, img2)
	##print(distance)
	##print(paths)
	##dtwvis.plot_warping(img1, img2, path, filename="warp.png")


'''