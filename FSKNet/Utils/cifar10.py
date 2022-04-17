from keras_resnet.models import ResNet34
from keras_resnet.models import ResNet18

# Training parameters

# adabound parameters

# Subtracting pixel mean improve accuracy

# Model version

n = 34
assert n in (18, 34), 'N must be 18 or 34'

depth = n
version = 1

# model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the cifar10 data

# Input image dimensions

# Normalize data

# If subtract pixel mean is enabled

# Convert class vectors to binary class matrices

def lr_schedule(epoch):
    pass

if n==18:
    model = ResNet18(input_shape=input_shape,depth=depth)
else:
    model = ResNet34(input_shape = input_shape,depth=depth)

model.compile(loss='categorical_crossentropy',optimizer=AdaBound())