from keras import models
from keras.preprocessing import image
import numpy as np
import os
# test data
print('---Load test data---')
test_data_name = "data_2000_"
test_data = np.load("data/" + test_data_name + ".npy")
mask_data_name = "data_mask_2000_"
mask_data = np.load("data/" + mask_data_name + ".npy")

test_data = test_data[1800:1810].astype('float') / 255
mask_data = mask_data[1800:1810].astype('float') / 255


# save mask image
print('---Save mask for comparing---')
mask_dir = "/home/xingyu/Desktop/unet-2/results/"+test_data_name+"mask/"
os.mkdir(mask_dir)

for i in range(np.shape(mask_data)[0]):
    img = mask_data[i]
    img = image.array_to_img(img)
    img.save(mask_dir + test_data_name + "_%d.tif" % i)

# load model
model_name = 'data_mask_2000_model_4'
model = models.load_model("model/"+model_name+".hdf5")

# evaluate the model and predict result
loss, acc = model.evaluate(test_data, mask_data, batch_size=4, verbose=1)
print("loss=", loss, "acc=", acc)
results = model.predict(test_data, batch_size=2)
np.save(mask_dir + model_name + "_predict.npy", results)
imgs = np.load(mask_dir + model_name + "_predict.npy")
print('---array to image ---')
for i in range(np.shape(imgs)[0]):
     img = imgs[i]
     img = image.array_to_img(img)
     img.save(mask_dir+"_predict_%d.tif" % i)

