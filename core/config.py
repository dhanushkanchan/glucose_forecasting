from easydict import EasyDict

C           				= EasyDict()
# Usage : from config import cfg
# Generic LSTM/GRU based configuration

cfg            				= C

# DATA
C.DATA  					= EasyDict()
C.DATA.PATH					= "./data/1131.csv"
C.DATA.VALIDATION_SPLIT		= 0.7
C.DATA.NUM_FEATURES			= 3
C.DATA.LAG					= 1
C.DATA.INPUT_TIMESTEPS		= 5
C.DATA.OUTPUT_TIMESTEPS		= 5

# TRAIN
C.TRAIN 					= EasyDict()
C.TRAIN.EPOCHS				= 50
C.TRAIN.BATCH_SIZE			= 24
C.TRAIN.LSTM_NEURONS		= 200
C.TRAIN.STATEFUL			= True
C.TRAIN.RESET_STATES		= True
C.TRAIN.LOSS 				= 'mse'
C.TRAIN.METRICS				= ['mae', 'mape']
C.TRAIN.OPTIMIZER 			= 'adam'
C.TRAIN.LOAD_MODEL			= ''
C.TRAIN.SAVE_MODEL 			= True
C.TRAIN.DROPOUT				= 0
C.TRAIN.EARLY_STOPPING		= True