import scipy

# mat = scipy.io.loadmat("../data/building models/models_multi_matrices.mat")
# models = mat["models_multi_matrices"]["models"][0, 0]
# model = models[0][18][0][0]

mat = scipy.io.loadmat("../data/building models/valid_models_indices.mat")
valid_index_list = mat["valid_models_indices"][:, 0]
print(1449 in valid_index_list)
