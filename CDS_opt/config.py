class Config(object):	
	apr_dir = './model/'
	data_dir = './data/'
	out_dir = './result/'
	model_name = 'checkpoints.pt'
	epoch = 5
	bert_model = 'bert-base-cased'
	lr = 5e-5
	eps = 1e-8
	batch_size = 1

	training_data = 'train_cdbox.txt'
	val_data = 'dev_cdbox.txt'
	test_data = 'test.txt'
	test_out = 'test_prediction.txt'


