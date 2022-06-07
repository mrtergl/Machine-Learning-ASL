from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range= 30,
        width_shift_range = 0.2,
        height_shift_range= 0.2,
        shear_range= 0.2,
        zoom_range = 0.2,
        horizontal_flip =True,
        fill_mode = 'nearest')

i = 0

for batch in datagen.flow_from_directory(directory = 'img/', batch_size = 400, target_size = (100,100),
                                             classes= 'Z',color_mode ='rgb', save_to_dir= 'img/Z', save_prefix= 'aug', save_format = 'png'):
    i +=1
    if i>0:
        break