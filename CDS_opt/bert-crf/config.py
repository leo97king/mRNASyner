class Config(object):	
	apr_dir = 'D:/bert_crf-master/model/'
	data_dir = 'D:/bert_crf-master/corpus/'
	model_name = 'model_4.pt'
	epoch = 5
	bert_model = 'bert-base-cased'
	lr = 5e-5
	eps = 1e-8
	batch_size = 1
	mode = 'train' # for prediction mode = "prediction"
	training_data = 'train_cdbox.txt'
	val_data = 'dev_cdbox.txt'
	test_data = 'test_cdbox.txt'
	test_out = 'test_prediction.txt'
	raw_prediction_output = 'raw_prediction.txt'

