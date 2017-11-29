# make sure  numpy, scipy, pandas, sklearn are installed, otherwise run
# pip install numpy scipy pandas scikit-learn
import numpy as np
import pandas as pd
from pprint import pprint
from scipy import io
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import CSVLogger, EarlyStopping
from custom import AdaptativeBiHyperbolic
from keras.layers import PReLU, ELU, LeakyReLU,ThresholdedReLU, Dropout
from keras import regularizers

def create_activation_layer(name):
    if name == 'abh':
        return AdaptativeBiHyperbolic()
    elif name == 'prelu':
        return PReLU()
    elif name == 'lrelu':
        return LeakyReLU()
    elif name == 'trelu':
        return ThresholdedReLU()
    elif name == 'elu':
        return ELU()
    elif name == 'relu':
        return Activation('relu')
    elif name == 'tanh':
        return Activation('tanh')

def get_model(activation_funtion, n_hidden_layers, hidden_size, lr, l2, dropout_rate):
    activation_layers = []
    model = Sequential()
    model.add(Dense(hidden_size,input_shape=(1644,),kernel_regularizer=regularizers.l2(l2)))
    activation_layer = create_activation_layer(activation_funtion)
    model.add(activation_layer)
    activation_layers.append(activation_layer)
    model.add(Dropout(dropout_rate))
    for l in range(n_hidden_layers):
        model.add(Dense(hidden_size, kernel_regularizer=regularizers.l2(l2)))
        activation_layer = create_activation_layer(activation_funtion)
        model.add(activation_layer)
        activation_layers.append(activation_layer)
        model.add(Dropout(dropout_rate))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()


    sgd = SGD(lr=l2, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    return model

#%%
# load data

y_tr = pd.read_csv('tox21/tox21_labels_train.csv.gz',
                   index_col=0, compression="gzip")
y_te = pd.read_csv('tox21/tox21_labels_test.csv.gz',
                   index_col=0, compression="gzip")
x_tr_dense = pd.read_csv('tox21/tox21_dense_train.csv.gz',
                         index_col=0, compression="gzip").values
x_te_dense = pd.read_csv('tox21/tox21_dense_test.csv.gz',
                         index_col=0, compression="gzip").values
x_tr_sparse = io.mmread('tox21/tox21_sparse_train.mtx.gz').tocsc()
x_te_sparse = io.mmread('tox21/tox21_sparse_test.mtx.gz').tocsc()

# filter out very sparse features
sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])

x_te.shape
#%%
# Build a random forest model for all twelve assays

aucs = []
for target in y_tr.columns:
    es = EarlyStopping(monitor='val_loss', patience=5)

    rows_tr = np.isfinite(y_tr[target]).values
    rows_te = np.isfinite(y_te[target]).values
    
    model = get_model('abh', 6, 1024, 0.01, 0.001, 0.05)

    y_train = np_utils.to_categorical(y_tr[target][rows_tr], 2)
    model.fit(x_tr[rows_tr], y_train, batch_size=128, epochs=100, validation_split=1/10, verbose=1, callbacks=[es])

    p_te = model.predict_proba(x_te[rows_te])
    auc_te = roc_auc_score(y_te[target][rows_te], p_te[:, 1])
    aucs.append(auc_te)

print('mean:', np.mean(aucs))
print('std:', np.std(aucs))
#print("%15s: %3.5f" % (target, auc_te))
pprint(aucs)
