import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from PIL import Image

# modelimizi yükledik
model = tf.keras.saving.load_model("./web/model/model.hdf5")

# class isimlerimizi değişkene atadık
class_names = ['at', 'fil', 'inek', 'kedi', 'kelebek', 'kopek', 'koyun', 'orumcek', 'sincap', 'tavuk']

# kullanıcıdan bir resim aldık
uploaded_img =  st.file_uploader("Lütfen resim yükleyiniz.", type=['png','jpg'])

if uploaded_img is not None:

    # streamlit yüklenen dosyaları UploadedFile olarak işlediği için,
    # öncelikle yüklediğimiz dosyayı bir resim olarak açtık ve bir değişkene atadık
    resized_img = Image.open(uploaded_img)

    # ardından resmimizin boytunu değiştirdik
    resized_img = resized_img.resize((224,224))

    # resmimizi ekranda gösterdik
    st.image(resized_img)

    # resmimizi numpy dizisine dönüştürdük
    resized_img = tf.keras.utils.img_to_array(resized_img)

    #
    resized_img = np.expand_dims(resized_img, axis=0)

    result = model.predict(resized_img)
    result = class_names[np.argmax(result[0])]

    # sonucu ekrana yazdırdık
    st.text("Bence, bu bir {}.".format(result))