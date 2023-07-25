import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from PIL import Image

st.set_page_config(
    page_title= "Animal Classification",
    page_icon= ":cat:",
    layout='centered'
)

#column yapımızı oluşturduk
col1,col2 = st.columns(2)

# modelimizi yükledik
model = tf.keras.saving.load_model("./web/model/model.hdf5")

# class isimlerimizi değişkene atadık
class_names = ['at', 'fil', 'inek', 'kedi', 'kelebek', 'kopek', 'koyun', 'orumcek', 'sincap', 'tavuk']

with st.sidebar:
    # kullanıcıdan bir resim aldık
    uploaded_img =  st.file_uploader("Lütfen resim yükleyiniz.", type=['png','jpg'])

if uploaded_img is not None:

    # streamlit yüklenen dosyaları UploadedFile olarak işlediği için,
    # öncelikle yüklediğimiz dosyayı bir resim olarak açtık ve bir değişkene atadık
    resized_img = Image.open(uploaded_img)

    # ardından resmimizin boytunu değiştirdik
    resized_img = resized_img.resize((224,224))

    # boyutlandırılmış resmimizi gösterdik
    with col1:
        st.image(resized_img)

    # resmimizi numpy dizisine dönüştürdük
    resized_img = tf.keras.utils.img_to_array(resized_img)

    #
    resized_img = np.expand_dims(resized_img, axis=0)

    # sonucumuzu aldık ve bir değişkene atadık
    result = model.predict(resized_img)
    result = class_names[np.argmax(result[0])]

    st.divider()

    # sonucu ekrana yazdırdık
    st.text("Bence, bu bir {}.".format(result))

    # sonuçtaki hayvana göre hayvan resmini gösterdik
    with col2:
        if result == "at":
            result_img = Image.open("images/at.jpeg")
            st.image(result_img.resize((224,224)))
        elif result == "fil":
            result_img = Image.open("images/fil.jpeg")
            st.image(result_img.resize((224,224)))
        elif result == "inek":
            result_img = Image.open("images/inek.jpeg")
            st.image(result_img.resize((224,224)))
        elif result == "kedi":
            result_img = Image.open("images/kedi.jpeg")
            st.image(result_img.resize((224,224)))
        elif result == "kelebek":
            result_img = Image.open("images/kelebek.png")
            st.image(result_img.resize((224,224)))
        elif result == "kopek":
            result_img = Image.open("images/kopek.jpeg")
            st.image(result_img.resize((224,224)))
        elif result == "koyun":
            result_img = Image.open("images/koyun.jpg")
            st.image(result_img.resize((224,224)))
        elif result == "orumcek":
            result_img = Image.open("images/orumcek.jpg")
            st.image(result_img.resize((224,224)))
        elif result == "sincap":
            result_img = Image.open("images/sincap.jpeg")
            st.image(result_img.resize((224,224)))
        else:
            result_img = Image.open("images/tavuk.jpeg")
            st.image(result_img.resize((224,224)))