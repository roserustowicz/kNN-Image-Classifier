import unpack_idx_format as unpack_idx
import numpy as np

# Read in MNIST dataset
tr_lbl, tr_img = unpack_idx.read(dataset = "training", path = "../MNIST-data");
te_lbl, te_img = unpack_idx.read(dataset = "testing", path = "../MNIST-data");

# Reshape the images so that they are [r c p], rather than [p r c]
tr_img = np.transpose(tr_img, (1, 2, 0))
te_img = np.transpose(te_img, (1, 2, 0))
[tr_r, tr_c, tr_p] = tr_img.shape
[te_r, te_c, te_p] = te_img.shape

# Debugging to see if image is in expected format
#print(tr_img.shape)
#print(te_img.shape)
#print(tr_lbl[0])
#print(tr_img[0])
#unpack_idx.show(tr_img[:,1:10,0])
#print(np.amax(tr_img))
#print(np.amin(tr_img))
#print(np.mean(tr_img))

# Reshape the images so they are stretched out in columns, and concatenated column by column
tr_img = np.reshape(tr_img, [tr_r*tr_c, tr_p])
te_img = np.reshape(te_img, [te_r*te_c, te_p])

# Debugging to see if image is in expected format (again)
#unpack_idx.show(tr_img)
#print(tr_img.shape)
#img_1 = tr_img[:,0]
#img_1 = np.reshape(img_1, [28, 28])
#unpack_idx.show(img_1)

# Normalize the data to zero mean and unit variance
tr_img = (tr_img - np.mean(tr_img)) / np.sqrt(np.var(tr_img))
te_img = (te_img - np.mean(te_img)) / np.sqrt(np.var(te_img))

# Debugging to see if the datasets are normalized to zero mean and unit variance
#print(np.mean(tr_img))
#print(np.var(tr_img))
#print(np.mean(te_img))
#print(np.var(te_img))

# Split into folds for cross-validation
num_folds = 5
for fold_idx in range(num_folds):
	split_value = tr_p / num_folds	
	cur_train = tr_img[:,fold_idx*split_value:(fold_idx+1)*split_value]
	print(fold_idx*split_value)
	print((fold_idx+1)*split_value)
	print('-------------')







