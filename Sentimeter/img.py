from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from flask import Flask, render_template,request
import tensorflow as tf
import numpy as np
import keras
import os 



class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('accuracy') > .98) & (logs.get('val_accuracy') > .9):
            print("Reached 98% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)


train_generator = train_datagen.flow_from_directory(

        'dataset/training/', 
        target_size=(200, 200),  
        batch_size=128,
        class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
        'dataset/validation/',  
        target_size=(200, 200), 
        batch_size=128,
        class_mode='binary')

train_generator.class_indices

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit(
      train_generator,
      steps_per_epoch=1,  
      epochs=20,
      validation_data = validation_generator,callbacks=[callbacks])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('img.html')


@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method=='POST':
        imgfile = request.files['img']
    
        if imgfile:
            filename = imgfile.filename
            path = os.path.join('static/uploads',filename)
            imgfile.save(path)
            img = image.load_img(path ,target_size=(200,200))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            prediction = model.predict(x)
            if prediction == 0:
                return render_template('img.html', prediction_text="Sorry for the inconvenience caused by our product. We'll connect you to the customer helpline",filename=filename)
            else:
                return render_template('img.html', prediction_text="We are glad you liked our product.",filename=filename)
        else:
            return render_template('img.html', prediction_text="No file")

    elif request.method=='GET':
        return render_template('img.html', prediction_text="Get Method")



if __name__ == "__main__":
    app.run(debug=True)

