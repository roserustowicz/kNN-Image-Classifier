import unpack_idx_format as unpack_idx
import numpy as np

def calculate_distances(tr_data, te_data):
	#print tr_data.shape, te_data.shape
	# Calculate the Euclidean distance between the vector sets of te_data (X)
	#  and tr_data (Y), using the eqn: D = sqrt( X_2 + Y_2 - 2*XdotY)
	# Calculate the value of tr_data squared along axis 1
	tr_data_2 = np.sum((tr_data*tr_data), axis=1)
	#print tr_data_2.shape
	# Repeat the vector from the line above by the number of te_data examples
	tr_data_2_rep = np.array([tr_data_2]*te_data.shape[0]).transpose()
	#print tr_data_2_rep.shape
	# Calculate the value of te_data squared along axis 1
	te_data_2 = np.sum((te_data*te_data), axis=1)
	#print te_data_2.shape
	# Repeat vector from above by number of tr_data examples, and take transpose
	te_data_2_rep = np.array([te_data_2]*tr_data.shape[0])
	#print te_data_2_rep.shape
	# Calculate the dot product of te_data and tr_data, a matrix multiplication
	tr_dot_te = np.dot(tr_data, te_data.transpose())
	#print tr_dot_te.shape
	# Put it all together and calculate the distance matrix
	dist_mat = np.sqrt((te_data_2_rep + tr_data_2_rep - 2*tr_dot_te))
	return dist_mat

def predict_labels(dist_mat, tr_lbl, k_value):
	predicted_lbls = []
	for col_idx in range(dist_mat.shape[1]):
		cur_dists = dist_mat[:,col_idx]
		lbls_sorted = [x for (y,x) in sorted(zip(cur_dists,tr_lbl))]
		top_k = np.bincount(lbls_sorted[:k_value])
		prediction = np.argmax(top_k)
		predicted_lbls.append(prediction)
		#print prediction
		#print tr_lbl
		#print cur_dists
		#print lbls_sort
		#input("Press Enter to go onto the next one.")
	predicted_lbls = np.asarray(predicted_lbls)
	#print predicted_lbls
	return predicted_lbls

def normalize_data(tr_img, te_img):
	# Normalize the data to zero mean and unit variance
	tr_img = (tr_img - np.mean(tr_img)) / np.sqrt(np.var(tr_img))
	te_img = (te_img - np.mean(te_img)) / np.sqrt(np.var(te_img))

	# Debugging to see if the datasets are normalized to zero mean and unit variance
	#print(np.mean(tr_img))
	#print(np.var(tr_img))
	#print(np.mean(te_img))
	#print(np.var(te_img))
	return tr_img, te_img

def cross_validation(num_folds, k_value, tr_img, tr_lbl):
	# Split into folds
	tr_img_folds = np.array_split(tr_img, num_folds)
	tr_lbl_folds = np.array_split(tr_lbl, num_folds)

	# For each fold, get the accuracy
	acc = []
	for val_idx in range(num_folds):
		cur_tr_img = []
		cur_tr_lbl = []
		for fold_idx in range(num_folds):
			if fold_idx == val_idx:
				cur_val_img = tr_img_folds[fold_idx]
				cur_val_lbl = tr_lbl_folds[fold_idx]
			else:
				cur_tr_img.extend(tr_img_folds[fold_idx])
				cur_tr_lbl.extend(tr_lbl_folds[fold_idx])
		cur_val_img = np.asarray(cur_val_img)
		cur_val_lbl = np.asarray(cur_val_lbl)
		cur_tr_img = np.asarray(cur_tr_img)
		cur_tr_lbl = np.asarray(cur_tr_lbl)
		# Calculate the distances! 
		#print(val_idx)
		#print(len(cur_val_img))
		#print(len(cur_val_lbl))
		#print(len(cur_tr_img))
		#print(len(cur_tr_lbl))
		#print(len(cur_val_img))
		#print('-------------')
		dist_mat = calculate_distances(cur_tr_img, cur_val_img)
		predicted_lbls = predict_labels(dist_mat,cur_tr_lbl,k_value)
		#print len(predicted_lbls)

		# Analysis of results
		num_correct = np.sum(predicted_lbls == cur_val_lbl)
		accuracy = float(num_correct) / len(predicted_lbls)
		acc.append(accuracy)
		print 'Fold %d for validation. Got %d / %d correct --> accuracy: %f' % (val_idx+1, num_correct, len(predicted_lbls), accuracy)
	avg_acc = np.sum(acc)/len(acc)
	std_acc = np.std(acc)
	print 'AVERAGE ACCURACY: %f' % (avg_acc)
	print('---------------------------------------')
	return avg_acc, std_acc

def reshape_and_crop(tr_img, tr_lbl, te_img, te_lbl, num_train, num_test):
	# Reshape the images so they are stretched out in columns, and concatenated column by column
	[tr_p, tr_r, tr_c] = tr_img.shape
	[te_p, te_r, te_c] = te_img.shape
	tr_img = np.reshape(tr_img, [tr_p, tr_r*tr_c])
	te_img = np.reshape(te_img, [te_p, te_r*te_c])

	# Only use a small portion of the data for a quicker implementation
	tr_img = tr_img[:num_train, :]
	te_img = te_img[:num_test, :]
	tr_lbl = tr_lbl[:num_train]
	te_lbl = te_lbl[:num_test]

	return tr_img, tr_lbl, te_img, te_lbl

def main():
	num_train=60000
	num_test=1000
	test = 1
	
	# Read in MNIST dataset
	tr_lbl, tr_img = unpack_idx.read(dataset = "training", path = "../MNIST-data");
	te_lbl, te_img = unpack_idx.read(dataset = "testing", path = "../MNIST-data");
	#print 'INITIALLY tr_lbl size: %s, tr_img size: %s, te_lbl size: %s, te_img size: %s' % (tr_lbl.shape, tr_img.shape, te_lbl.shape, te_img.shape)

	# Reshape and crop the input 
	tr_img, tr_lbl, te_img, te_lbl = reshape_and_crop(tr_img, tr_lbl, te_img, te_lbl, num_train, num_test)
	#print 'AFTER RESHAPE AND CROP tr_lbl size: %s, tr_img size: %s, te_lbl size: %s, te_img size: %s' % (tr_lbl.shape, tr_img.shape, te_lbl.shape, te_img.shape)

	# Normalize the images with zero mean and unit variance
	tr_img, te_img = normalize_data(tr_img, te_img)
	#print 'AFTER NORMALIZATION tr_img size: %s, te_img size: %s' % (tr_img.shape, te_img.shape)

	if test == 0:
		acc_vals = []
		std_vals = []
		k_vals = []
		num_folds_vals = []
		for num_folds in [3, 5, 10]:
			for k_value in [1, 3, 5, 10, 20, 50, 100]:
				print 'NUM FOLDS: %d , K VALUE: %d' % (num_folds, k_value)
				avg_acc, std_acc = cross_validation(num_folds, k_value, tr_img, tr_lbl)
				acc_vals.append(avg_acc)
				std_vals.append(std_acc)
				k_vals.append(k_value)
				num_folds_vals.append(num_folds)

		f = open('my_results.txt', 'w')
		f.write('ACCURACIES: \n')
		for a in acc_vals:
			f.write(str(a) + '\n')
		f.write('\n')

		f.write('STDDEVs: \n')
		for s in std_vals:
			f.write(str(s) + '\n')
		f.write('\n')

		f.write('K VALUES: \n')
		for k in k_vals:
			f.write(str(k) + '\n')
		f.write('\n')

		f.write('NUMBER OF FOLDS: \n') 
		for nf in num_folds_vals:
			f.write(str(nf) + '\n')
		f.close()
	
	else:
		k_value = 1
		dist_mat = calculate_distances(tr_img, te_img)
		predicted_lbls = predict_labels(dist_mat,tr_lbl,k_value)
		#print len(predicted_lbls)
		# Analysis of results
		num_correct = np.sum(predicted_lbls == te_lbl)
		accuracy = float(num_correct) / len(predicted_lbls)
		print 'Got %d / %d correct --> accuracy: %f' % (num_correct, len(predicted_lbls), accuracy)

if __name__ == '__main__':
	main()
