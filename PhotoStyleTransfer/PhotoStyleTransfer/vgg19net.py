#  To build the vgg19 net and other functions

import scipy.io as sio
import numpy as np
import tensorflow as tf

CONTENT_LAYERS=[('input',1.)]
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
class vgg:
	# The path of vgg19 weights
	__VGG19_PATH='Photo Style Transfer/VGG19_mat/imagenet-vgg-verydeep-19.mat'
	# The dict of the vgg19 weights
	__param={}
	net={}

	def __init__(self,mat_path=__VGG19_PATH):
		# Load VGG19 net
		param=sio.loadmat(mat_path)
		layers=param['layers'][0]

		# layer 1 in VGG
		self.__param['filter1_1'],self.__param['bias1_1']=layers[0][0][0][0][0]  # filter shape: (3,3,3,64)
		self.__param['filter1_1']=np.transpose(self.__param['filter1_1'],(1,0,2,3))

		self.__param['filter1_2'],self.__param['bias1_2']=layers[2][0][0][0][0]  # filter shape: (3,3,3,64)
		self.__param['filter1_2']=np.transpose(self.__param['filter1_2'],(1,0,2,3))

		# layer 2 in VGG
		self.__param['filter2_1'],self.__param['bias2_1']=layers[5][0][0][0][0]  # filter shape: (3,3,64,128)
		self.__param['filter2_1']=np.transpose(self.__param['filter2_1'],(1,0,2,3))

		self.__param['filter2_2'],self.__param['bias2_2']=layers[7][0][0][0][0]  # filter shape: (3,3,128,128)
		self.__param['filter2_2']=np.transpose(self.__param['filter2_2'],(1,0,2,3))

		# layer 3 in VGG
		self.__param['filter3_1'],self.__param['bias3_1']=layers[10][0][0][0][0]  # filter shape: (3,3,128,256)
		self.__param['filter3_1']=np.transpose(self.__param['filter3_1'],(1,0,2,3))

		self.__param['filter3_2'],self.__param['bias3_2']=layers[12][0][0][0][0]  # filter shape: (3,3,256,256)
		self.__param['filter3_2']=np.transpose(self.__param['filter3_2'],(1,0,2,3))

		self.__param['filter3_3'],self.__param['bias3_3']=layers[14][0][0][0][0]  # filter shape: (3,3,256,256)
		self.__param['filter3_3']=np.transpose(self.__param['filter3_3'],(1,0,2,3))

		self.__param['filter3_4'],self.__param['bias3_4']=layers[16][0][0][0][0]  # filter shape: (3,3,256,256)
		self.__param['filter3_4']=np.transpose(self.__param['filter3_4'],(1,0,2,3))

		# layer 4 in VGG
		self.__param['filter4_1'],self.__param['bias4_1']=layers[19][0][0][0][0]  # filter shape: (3,3,256,512)
		self.__param['filter4_1']=np.transpose(self.__param['filter4_1'],(1,0,2,3))

		self.__param['filter4_2'],self.__param['bias4_2']=layers[21][0][0][0][0]  # filter shape: (3,3,512,512)
		self.__param['filter4_2']=np.transpose(self.__param['filter4_2'],(1,0,2,3))

		self.__param['filter4_3'],self.__param['bias4_3']=layers[23][0][0][0][0]  # filter shape: (3,3,512,512)
		self.__param['filter4_3']=np.transpose(self.__param['filter4_3'],(1,0,2,3))

		self.__param['filter4_4'],self.__param['bias4_4']=layers[25][0][0][0][0]  # filter shape: (3,3,512,512)
		self.__param['filter4_4']=np.transpose(self.__param['filter4_4'],(1,0,2,3))

		# layer 5 in VGG
		self.__param['filter5_1'],self.__param['bias5_1']=layers[28][0][0][0][0]  # filter shape: (3,3,512,512)
		self.__param['filter5_1']=np.transpose(self.__param['filter5_1'],(1,0,2,3))

		self.__param['filter5_2'],self.__param['bias5_2']=layers[30][0][0][0][0]  # filter shape: (3,3,512,512)
		self.__param['filter5_2']=np.transpose(self.__param['filter5_2'],(1,0,2,3))

		self.__param['filter5_3'],self.__param['bias5_3']=layers[32][0][0][0][0]  # filter shape: (3,3,512,512)
		self.__param['filter5_3']=np.transpose(self.__param['filter5_3'],(1,0,2,3))

		self.__param['filter5_4'],self.__param['bias5_4']=layers[34][0][0][0][0]  # filter shape: (3,3,512,512)
		self.__param['filter5_4']=np.transpose(self.__param['filter5_4'],(1,0,2,3))

	# max_pool layer in vgg19
	def __pool_layer(self,tf_pre_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'):
		return tf.nn.max_pool(tf_pre_layer,ksize=ksize,strides=strides,padding=padding)

	# conv layer in vgg19
	def __conv_layer(self,tf_pre_layer,tf_filter,tf_bias,strides=[1,1,1,1],padding='SAME'):
		conv=tf.nn.conv2d(tf_pre_layer,tf_filter,strides=strides,padding=padding)
		return tf.nn.relu(tf.add(conv,tf_bias))

	def generate_net(self,tf_image):
		self.net['input']=tf_image
		# layer 1
		self.net['relu1_1']=self.__conv_layer(tf_image,tf.constant(self.__param['filter1_1']),tf.constant(self.__param['bias1_1']))
		self.net['relu1_2']=self.__conv_layer(self.net['relu1_1'],tf.constant(self.__param['filter1_2']),tf.constant(self.__param['bias1_2']))
		# layer 2
		self.net['relu2_1']=self.__conv_layer(self.__pool_layer(self.net['relu1_2']),tf.constant(self.__param['filter2_1']),tf.constant(self.__param['bias2_1']))
		self.net['relu2_2']=self.__conv_layer(self.net['relu2_1'],tf.constant(self.__param['filter2_2']),tf.constant(self.__param['bias2_2']))
		# layer 3
		self.net['relu3_1']=self.__conv_layer(self.__pool_layer(self.net['relu2_2']),tf.constant(self.__param['filter3_1']),tf.constant(self.__param['bias3_1']))
		self.net['relu3_2']=self.__conv_layer(self.net['relu3_1'],tf.constant(self.__param['filter3_2']),tf.constant(self.__param['bias3_2']))
		self.net['relu3_3']=self.__conv_layer(self.net['relu3_2'],tf.constant(self.__param['filter3_3']),tf.constant(self.__param['bias3_3']))
		self.net['relu3_4']=self.__conv_layer(self.net['relu3_3'],tf.constant(self.__param['filter3_4']),tf.constant(self.__param['bias3_4']))
		# layer 4
		self.net['relu4_1']=self.__conv_layer(self.__pool_layer(self.net['relu3_4']),tf.constant(self.__param['filter4_1']),tf.constant(self.__param['bias4_1']))
		self.net['relu4_2']=self.__conv_layer(self.net['relu4_1'],tf.constant(self.__param['filter4_2']),tf.constant(self.__param['bias4_2']))
		self.net['relu4_3']=self.__conv_layer(self.net['relu4_2'],tf.constant(self.__param['filter4_3']),tf.constant(self.__param['bias4_3']))
		self.net['relu4_4']=self.__conv_layer(self.net['relu4_3'],tf.constant(self.__param['filter4_4']),tf.constant(self.__param['bias4_4']))
		# layer 5
		self.net['relu5_1']=self.__conv_layer(self.__pool_layer(self.net['relu4_4']),tf.constant(self.__param['filter5_1']),tf.constant(self.__param['bias5_1']))
		self.net['relu5_2']=self.__conv_layer(self.net['relu5_1'],tf.constant(self.__param['filter5_2']),tf.constant(self.__param['bias5_2']))
		self.net['relu5_3']=self.__conv_layer(self.net['relu5_2'],tf.constant(self.__param['filter5_3']),tf.constant(self.__param['bias5_3']))

		self.net['relu5_4']=self.__conv_layer(self.net['relu5_3'],tf.constant(self.__param['filter5_4']),tf.constant(self.__param['bias5_4']))

		return self.net

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
def get_loss_fun(tf_cnt_img,tf_sty_img,tf_nis_img):
	# build vgg net
	content_net=vgg()
	result_net=vgg()
	style_net=vgg()
	content_net.generate_net(tf_cnt_img)
	style_net.generate_net(tf_sty_img)
	result_net.generate_net(tf_nis_img)

	style_loss=0
	content_loss=0

	sess=tf.Session()
	init=tf.global_variables_initializer()
	sess.run(init)

	# get content loss
	for name,ratio in CONTENT_LAYERS:
		# get standard content feature
		std_cnt=sess.run(content_net.net[name])
		rst_cnt=result_net.net[name]
		content_loss+=content_loss_per_feature(std_cnt,rst_cnt,ratio)

	# get style loss
	for name,ratio in STYLE_LAYERS:
		# get standard style feature
		std_sty=sess.run(style_net.net[name])
		rst_sty=result_net.net[name]
		style_loss+=style_loss_per_feature(std_sty,rst_sty,ratio)

	return content_loss,style_loss