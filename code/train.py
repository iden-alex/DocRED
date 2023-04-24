import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'BiLSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--batch_size', type = int, default = 40)
parser.add_argument('--max_epoch', type = int, default = 200)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--na_triple_coef', type = int, default = 5)



args = parser.parse_args()
model = {
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
}

con = config.Config(args)
con.set_batch_size(args.batch_size)
con.set_lr(args.lr)
con.set_max_epoch(args.max_epoch)
con.set_na_triple_coef(args.na_triple_coef)

con.load_train_data()
con.load_test_data()
# con.set_train_model()
print("start train")
con.train(model[args.model_name], args.save_name)
