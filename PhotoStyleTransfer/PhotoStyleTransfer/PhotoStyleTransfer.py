import tensorflow as tf
import numpy as np
import cv2 as cv
import vgg19net as vn
import time

CONTENT_PATH='Photo Style Transfer/Content/tower.jpg'
STYLE_PATH='Photo Style Transfer/Style/sky.jpg'
RESULT_PATH='Photo Style Transfer/Result/'

IMG_WIDTH=650
IMG_HEIGHT=450
NOISE_RATIO=0.5

STYLE_WEIGHT=0
CONTENT_WEIGHT=10

LEARNING_RATE=1
EPOCH=1000

#  mean value used in VGG19-mat
# mean = param['normalization'][0][0][0]
MEAN_VAL=np.reshape([123.68,116.779,103.939],(1,1,1,3))

#  Load content image
img_cnt=cv.imread(CONTENT_PATH)
img_cnt=cv.resize(img_cnt,(IMG_WIDTH,IMG_HEIGHT),interpolation=cv.INTER_LINEAR)
'''
#  X=Change channel BGR to RGB
R=img_cnt[:,:,2]
img_cnt[:,:,2]=img_cnt[:,:,0]
img_cnt[:,:,0]=R
'''
img_cnt=np.reshape(img_cnt,(1,IMG_HEIGHT,IMG_WIDTH,3))
img_cnt=img_cnt-MEAN_VAL

#  Load style image
img_sty=cv.imread(STYLE_PATH)
img_sty=cv.resize(img_sty,(IMG_WIDTH,IMG_HEIGHT),interpolation=cv.INTER_LINEAR)
'''
#  X=Change channel BGR to RGB
R=img_sty[:,:,2]
img_sty[:,:,2]=img_sty[:,:,0]
img_sty[:,:,0]=R
'''
img_sty=np.reshape(img_sty,(1,IMG_HEIGHT,IMG_WIDTH,3))
img_sty=img_sty-MEAN_VAL

#  Generate noise image

nis_img=np.random.uniform(-20,20,(1, IMG_HEIGHT,IMG_WIDTH,3)).astype('float32')
nis_img=nis_img*NOISE_RATIO+img_cnt*(1-NOISE_RATIO)
result_img=tf.Variable(nis_img)
nis_img=tf.cast(result_img,tf.float32)
'''
result_img=tf.Variable(tf.truncated_normal((1, IMG_HEIGHT,IMG_WIDTH,3),stddev=20))
nis_img=result_img
'''


#cnt_loss,sty_loss=vn.get_loss_fun(tf.cast(tf.constant(img_cnt),tf.float32),tf.cast(tf.constant(img_sty),tf.float32),nis_img)

#cost=CONTENT_WEIGHT*cnt_loss+STYLE_WEIGHT*sty_loss

result_vgg=vn.vgg()
content_vgg=vn.vgg()
net1=result_vgg.generate_net(nis_img)
net2=content_vgg.generate_net(tf.cast(img_cnt,tf.float32))
sess=tf.Session()
sess.run(tf.global_variables_initializer())
cost=tf.nn.l2_loss(nis_img-net2['input'])

train=tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

init=tf.global_variables_initializer()
t0=time.time()
with tf.Session() as sess:
	sess.run(init)
	for i in range(0,EPOCH):
		_,_cost,_img=sess.run([train,cost,result_img])
		print(_img[0][0][0])
		if (i+1)%10==0:
			print('Step:',i+1,', Loss:',_cost,', Time:',time.time()-t0)
			_img+=MEAN_VAL
			_img=np.clip(_img,0,255).astype(np.uint8)
			_img=np.reshape(_img,(IMG_HEIGHT,IMG_WIDTH,3))
			cv.imshow('Rao',_img)
			cv.waitKey(10)