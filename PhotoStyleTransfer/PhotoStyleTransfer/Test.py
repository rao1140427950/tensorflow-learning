#  To build the vgg19 net and other functions

import scipy.io as sio
import numpy as np
import tensorflow as tf
import cv2 as cv
import vgg19net as vn
import time

CONTENT_PATH='Photo Style Transfer/Content/path.jpg'
STYLE_PATH='Photo Style Transfer/Style/style1.jpg'
RESULT_PATH='Photo Style Transfer/Result/'
EXISTING_STEP_PATH=RESULT_PATH+'Tower_stars_step4000.jpg'

IMG_WIDTH=650
IMG_HEIGHT=450
NOISE_RATIO=0.25

STYLE_WEIGHT=500
CONTENT_WEIGHT=20

LEARNING_RATE=0.3
LEARNING_RATE_DECAY=0.999
START_STEP=0
EPOCH=4000

#  mean value used in VGG19-mat
# mean = param['normalization'][0][0][0]
MEAN_VAL=np.reshape([123.68,116.779,103.939],(1,1,1,3))

CONTENT_LAYERS=[('relu4_2',1.)]
STYLE_LAYERS=[('relu1_1',0.2), ('relu2_1',0.2),('relu3_1',0.2),('relu4_1',0.2),('relu5_1',0.2)]

'''
layers = ( 
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4','pool3', 
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5' 
    ) 
'''

# vgg19 net

# The path of vgg19 weights
__VGG19_PATH='Photo Style Transfer/VGG19_mat/imagenet-vgg-verydeep-19.mat'
# The dict of the vgg19 weights
__param={}

def init_vggparam(mat_path=__VGG19_PATH):
	# Load VGG19 net
	param=sio.loadmat(mat_path)
	layers=param['layers'][0]

	# layer 1 in VGG
	__param['filter1_1'],__param['bias1_1']=layers[0][0][0][0][0]  # filter shape: (3,3,3,64)
	__param['filter1_1']=np.transpose(__param['filter1_1'],(1,0,2,3))

	__param['filter1_2'],__param['bias1_2']=layers[2][0][0][0][0]  # filter shape: (3,3,3,64)
	__param['filter1_2']=np.transpose(__param['filter1_2'],(1,0,2,3))

	# layer 2 in VGG
	__param['filter2_1'],__param['bias2_1']=layers[5][0][0][0][0]  # filter shape: (3,3,64,128)
	__param['filter2_1']=np.transpose(__param['filter2_1'],(1,0,2,3))

	__param['filter2_2'],__param['bias2_2']=layers[7][0][0][0][0]  # filter shape: (3,3,128,128)
	__param['filter2_2']=np.transpose(__param['filter2_2'],(1,0,2,3))

	# layer 3 in VGG
	__param['filter3_1'],__param['bias3_1']=layers[10][0][0][0][0]  # filter shape: (3,3,128,256)
	__param['filter3_1']=np.transpose(__param['filter3_1'],(1,0,2,3))

	__param['filter3_2'],__param['bias3_2']=layers[12][0][0][0][0]  # filter shape: (3,3,256,256)
	__param['filter3_2']=np.transpose(__param['filter3_2'],(1,0,2,3))

	__param['filter3_3'],__param['bias3_3']=layers[14][0][0][0][0]  # filter shape: (3,3,256,256)
	__param['filter3_3']=np.transpose(__param['filter3_3'],(1,0,2,3))

	__param['filter3_4'],__param['bias3_4']=layers[16][0][0][0][0]  # filter shape: (3,3,256,256)
	__param['filter3_4']=np.transpose(__param['filter3_4'],(1,0,2,3))

	# layer 4 in VGG
	__param['filter4_1'],__param['bias4_1']=layers[19][0][0][0][0]  # filter shape: (3,3,256,512)
	__param['filter4_1']=np.transpose(__param['filter4_1'],(1,0,2,3))

	__param['filter4_2'],__param['bias4_2']=layers[21][0][0][0][0]  # filter shape: (3,3,512,512)
	__param['filter4_2']=np.transpose(__param['filter4_2'],(1,0,2,3))

	__param['filter4_3'],__param['bias4_3']=layers[23][0][0][0][0]  # filter shape: (3,3,512,512)
	__param['filter4_3']=np.transpose(__param['filter4_3'],(1,0,2,3))

	__param['filter4_4'],__param['bias4_4']=layers[25][0][0][0][0]  # filter shape: (3,3,512,512)
	__param['filter4_4']=np.transpose(__param['filter4_4'],(1,0,2,3))

	# layer 5 in VGG
	__param['filter5_1'],__param['bias5_1']=layers[28][0][0][0][0]  # filter shape: (3,3,512,512)
	__param['filter5_1']=np.transpose(__param['filter5_1'],(1,0,2,3))

	__param['filter5_2'],__param['bias5_2']=layers[30][0][0][0][0]  # filter shape: (3,3,512,512)
	__param['filter5_2']=np.transpose(__param['filter5_2'],(1,0,2,3))

	__param['filter5_3'],__param['bias5_3']=layers[32][0][0][0][0]  # filter shape: (3,3,512,512)
	__param['filter5_3']=np.transpose(__param['filter5_3'],(1,0,2,3))

	__param['filter5_4'],__param['bias5_4']=layers[34][0][0][0][0]  # filter shape: (3,3,512,512)
	__param['filter5_4']=np.transpose(__param['filter5_4'],(1,0,2,3))

# max_pool layer in vgg19
def __pool_layer(tf_pre_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'):
	return tf.nn.max_pool(tf_pre_layer,ksize=ksize,strides=strides,padding=padding)

# conv layer in vgg19
def __conv_layer(tf_pre_layer,tf_filter,tf_bias,strides=[1,1,1,1],padding='SAME'):
	conv=tf.nn.conv2d(tf_pre_layer,tf_filter,strides=strides,padding=padding)
	return tf.nn.relu(tf.add(conv,tf_bias))

def generate_net(tf_image):
	net={}

	net['input']=tf_image
	# layer 1
	net['relu1_1']=__conv_layer(tf_image,tf.constant(__param['filter1_1']),tf.constant(__param['bias1_1']))
	net['relu1_2']=__conv_layer(net['relu1_1'],tf.constant(__param['filter1_2']),tf.constant(__param['bias1_2']))
	# layer 2
	net['relu2_1']=__conv_layer(__pool_layer(net['relu1_2']),tf.constant(__param['filter2_1']),tf.constant(__param['bias2_1']))
	net['relu2_2']=__conv_layer(net['relu2_1'],tf.constant(__param['filter2_2']),tf.constant(__param['bias2_2']))
	# layer 3
	net['relu3_1']=__conv_layer(__pool_layer(net['relu2_2']),tf.constant(__param['filter3_1']),tf.constant(__param['bias3_1']))
	net['relu3_2']=__conv_layer(net['relu3_1'],tf.constant(__param['filter3_2']),tf.constant(__param['bias3_2']))
	net['relu3_3']=__conv_layer(net['relu3_2'],tf.constant(__param['filter3_3']),tf.constant(__param['bias3_3']))
	net['relu3_4']=__conv_layer(net['relu3_3'],tf.constant(__param['filter3_4']),tf.constant(__param['bias3_4']))
	# layer 4
	net['relu4_1']=__conv_layer(__pool_layer(net['relu3_4']),tf.constant(__param['filter4_1']),tf.constant(__param['bias4_1']))
	net['relu4_2']=__conv_layer(net['relu4_1'],tf.constant(__param['filter4_2']),tf.constant(__param['bias4_2']))
	net['relu4_3']=__conv_layer(net['relu4_2'],tf.constant(__param['filter4_3']),tf.constant(__param['bias4_3']))
	net['relu4_4']=__conv_layer(net['relu4_3'],tf.constant(__param['filter4_4']),tf.constant(__param['bias4_4']))
	# layer 5
	net['relu5_1']=__conv_layer(__pool_layer(net['relu4_4']),tf.constant(__param['filter5_1']),tf.constant(__param['bias5_1']))
	net['relu5_2']=__conv_layer(net['relu5_1'],tf.constant(__param['filter5_2']),tf.constant(__param['bias5_2']))
	net['relu5_3']=__conv_layer(net['relu5_2'],tf.constant(__param['filter5_3']),tf.constant(__param['bias5_3']))

	net['relu5_4']=__conv_layer(net['relu5_3'],tf.constant(__param['filter5_4']),tf.constant(__param['bias5_4']))

	return net

# content loss in each feature map
def content_loss_per_feature(np_std_cnt,tf_rst_img,ratio):
	M=np_std_cnt.shape[1]*np_std_cnt.shape[2]
	N=np_std_cnt.shape[3]
	# l2 norm loss
	loss=(ratio/(2*N*M))*tf.nn.l2_loss(tf.subtract(tf_rst_img,tf.constant(np_std_cnt)))
	return loss

# calc the gram matrix
def gram_matrix(tf_mat,length,depth):
	gmat=tf.reshape(tf_mat,(length,depth))
	gmat=tf.matmul(tf.transpose(gmat),gmat)
	return gmat


# style loss in each feature map
def style_loss_per_feature(np_std_sty,tf_rst_img,ratio):
	M=np_std_sty.shape[1]*np_std_sty.shape[2]
	N=np_std_sty.shape[3]
	# calc gram matrix
	std_gram=gram_matrix(tf.constant(np_std_sty),M,N)
	rst_gram=gram_matrix(tf_rst_img,M,N)
	# l2 norm loss
	loss=(ratio/(4*(N**2)*(M**2)))*tf.nn.l2_loss(tf.subtract(std_gram,rst_gram))
	return loss


# Function to get the content loss function
def get_loss_fun(cnt_net,sty_net,rst_net):
	style_loss=0
	content_loss=0

	sess=tf.Session()
	init=tf.global_variables_initializer()
	sess.run(init)

	# get content loss
	for name,ratio in CONTENT_LAYERS:
		# get standard content feature
		std_cnt=sess.run(cnt_net[name])
		rst_cnt=rst_net[name]
		content_loss+=content_loss_per_feature(std_cnt,rst_cnt,ratio)

	# get style loss
	for name,ratio in STYLE_LAYERS:
		# get standard style feature
		std_sty=sess.run(sty_net[name])
		rst_sty=rst_net[name]
		style_loss+=style_loss_per_feature(std_sty,rst_sty,ratio)

	return content_loss,style_loss


#  Load style or content image
def load_img(path):
	img=cv.imread(path)
	img=cv.resize(img,(IMG_WIDTH,IMG_HEIGHT),interpolation=cv.INTER_LINEAR)
	img=np.reshape(img,(1,IMG_HEIGHT,IMG_WIDTH,3))
	img=img-MEAN_VAL
	return img

#  Init output image
FROM_CONTENT=0  # Init from content image: simply add some noise to the content image
FROM_NOISE=1  # Random init by truncated_normal
FROM_EXISTING_STEP=2  # Init from existing step and continue
def init_output_image(mod=FROM_CONTENT,path=None,cnt_image=None):
	if mod==FROM_CONTENT:
		nis_img=np.random.uniform(-20,20,(1, IMG_HEIGHT,IMG_WIDTH,3)).astype('float32')
		nis_img=nis_img*NOISE_RATIO+cnt_image*(1-NOISE_RATIO)
		nis_img=tf.Variable(nis_img)
		nis_img=tf.cast(nis_img,tf.float32)
		return nis_img

	if mod==FROM_NOISE:
		nis_img=tf.Variable(tf.truncated_normal((1, IMG_HEIGHT,IMG_WIDTH,3),stddev=20))
		return nis_img

	if mod==FROM_EXISTING_STEP:
		nis_img=load_img(path)
		nis_img=tf.Variable(nis_img)
		nis_img=tf.cast(nis_img,tf.float32)
		return nis_img


# Main frame
if __name__=='__main__':
	#  Load content image
	img_cnt=load_img(CONTENT_PATH)
	#  Load style image
	img_sty=load_img(STYLE_PATH)

	#  Generate noise image
	#result_img=init_output_image(mod=FROM_EXISTING_STEP,path=EXISTING_STEP_PATH)
	result_img=init_output_image(mod=FROM_CONTENT,cnt_image=img_cnt)
	
	init_vggparam()

	rst_net=generate_net(result_img)
	cnt_net=generate_net(tf.cast(img_cnt,tf.float32))
	sty_net=generate_net(tf.cast(img_sty,tf.float32))
	
	#cost=tf.nn.l2_loss(net1['relu4_2']-net2['relu4_2'])

	cnt_loss,sty_loss=get_loss_fun(cnt_net,sty_net,rst_net)
	cost=CONTENT_WEIGHT*cnt_loss+STYLE_WEIGHT*sty_loss
	
	global_step=tf.Variable(0,trainable=False)
	#learning_rate=tf.train.exponential_decay(LEARNING_RATE,global_step,int(1),LEARNING_RATE_DECAY)
	learning_rate=tf.constant(LEARNING_RATE)
	train=tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

	init=tf.global_variables_initializer()
	t0=time.time()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(START_STEP,START_STEP+EPOCH):
			_,_cost,_img,_lrn_rate=sess.run([train,cost,result_img,learning_rate])
			#print(_img[0][0][0])
			if (i+1)%5==0:
				print('Step:',i+1,', Loss:',_cost,', Learning rate:',_lrn_rate,', Time:',time.time()-t0)
				_img+=MEAN_VAL
				_img=np.clip(_img,0,255).astype(np.uint8)
				_img=np.reshape(_img,(IMG_HEIGHT,IMG_WIDTH,3))
				cv.imshow('Rao',_img)
				cv.waitKey(50)
				if (i+1)%500==0:
					cv.imwrite(RESULT_PATH+'path_style1_step'+str(i+1)+'.jpg',_img)