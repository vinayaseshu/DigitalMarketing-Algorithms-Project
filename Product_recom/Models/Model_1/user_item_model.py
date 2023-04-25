# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import itertools
import implicit

#ignore warning 
import warnings
warnings.filterwarnings("ignore")

#User_item Data
user_item = pd.read_csv('https://raw.githubusercontent.com/ashwinkadam/DigitalMarketing-Algorithms-Project/main/Product_recom/Data_Prep/master_data.csv')
user_item = user_item[['user_id','product_id','Quantity']]

print('Creatig CSR Matrix')
####################### CSR Matrix Creation ######################################

users = list(np.sort(user_item.user_id.unique()))
items = list(np.sort(user_item.product_id.unique()))
Quantity = list(user_item.Quantity)

# create zero-based index position <-> user/item ID mappings
index_to_user = pd.Series(users)
# create reverse mappings from user/item ID to index positions
user_to_index = pd.Series(data=index_to_user.index , index=index_to_user.values)

# Get the rows and columns for our new matrix
products_rows = user_item.product_id.astype(int)
users_cols = user_item.user_id.astype(int)

# Create a sparse matrix for our users and products containing number of purchases
sparse_product_user = sparse.csr_matrix((Quantity, (products_rows, users_cols)))
sparse_product_user.data = np.nan_to_num(sparse_product_user.data, copy=False)

sparse_user_product = sparse.csr_matrix((Quantity, (users_cols, products_rows)))
sparse_user_product.data = np.nan_to_num(sparse_user_product.data, copy=False)

print('CSR Matrix Created')
print('    ')
####################### User Item Model ########################################

print('Building Model......')
# initialize model
model = implicit.als.AlternatingLeastSquares(factors=100,
                                             regularization=0.05,
                                             iterations=20,
                                             num_threads=1)

alpha_val = 40
train_set = (sparse_product_user * alpha_val).astype('double')

# train the model
model.fit(train_set, show_progress = True)

print('    ')
####################### Save factors ########################################
print('Saving factors and index.........')
# Save the array to a file
np.save('user_factors.npy', model.user_factors)

# Save the array to a file
np.save('item_factors.npy', model.item_factors)

#Save the user_to_index to a file
user_to_index.to_csv('user_to_index.csv')

print('Model Built Successfully!!!!!!!')