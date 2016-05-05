import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import theano
import theano.tensor as T
import lasagne

# Read in the csv file and parse the first column as datetime objects for later calculation
train_df = pd.read_csv('data/train.csv', parse_dates=[0])
train_df = train_df[train_df['Y'] < 40]

# Create new features in the dataset for the year, month, day, and hour, but 
# treat them as categorical values (strings).
def date_extract(dt):
    return dt.strftime('%y'), dt.strftime('%m'), dt.strftime('%d'), \
           dt.strftime('%H'), dt.strftime('%M')

train_df['Year'], train_df['Month'], train_df['Day'], train_df['Hour'], \
    train_df['Minute'] = zip(*train_df['Dates'].map(date_extract))
    
train_df['X2'] = train_df['X']**2
train_df['XY'] = train_df['X']*train_df['Y']
train_df['Y2'] = train_df['Y']**2

# map category strings onto integers
cat_dict = {}
for idx, cat in enumerate(train_df['Category'].unique()):
    cat_dict[cat] = idx

train_df['Category'] = train_df['Category'].map(cat_dict)

y = train_df['Category'].values.astype(np.uint8)
X = pd.get_dummies(train_df[['X', 'Y', 'X2', 'XY', 'Y2', 'Year', 'Month', \
    'Day', 'Hour', 'Minute']]).values.astype(theano.config.floatX)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
    random_state=0)
sc = StandardScaler()
X_train[:,:5] = sc.fit_transform(X_train[:,:5])
X_test[:,:5] = sc.transform(X_test[:,:5])

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

input_var = T.matrix('inputs')
target_var = T.ivector('targets')

N_ROWS, N_COLS = X_train.shape
N_CATS = len(np.unique(y_train))
N_EPOCHS = 200

# Build mlp network
l_in = lasagne.layers.InputLayer(shape=(None, N_COLS), input_var=input_var)
#l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=10)
#l_hid_drop1 = lasagne.layers.DropoutLayer(l_hid1, p=0.25)
l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=15)
#l_hid_drop2 = lasagne.layers.DropoutLayer(l_hid2, p=0.25)
l_out = lasagne.layers.DenseLayer(l_hid2, num_units=N_CATS, \
                                  nonlinearity=lasagne.nonlinearities.softmax)

prediction = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, \
                                            momentum=0.9)

test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, \
                                                        target_var)
test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), \
                  dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

print('Training neural network to categorize\n \
 {} unique categories\n'.format(N_CATS))

for epoch in range(N_EPOCHS):
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, N_EPOCHS, \
                                                   time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1

print('\nFinal results:')
print('  test loss:\t\t\t{:.6f}'.format(test_err / test_batches))
print('  test accuracy:\t\t{:.2f} %'.format(test_acc / test_batches * 100))
