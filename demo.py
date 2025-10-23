from xml.parsers.expat import model
import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from tensorboard import program
import random
from sklearn.model_selection import train_test_split

import datetime
from sklearn.preprocessing import MinMaxScaler
from skimage.segmentation import mark_boundaries
from lime import lime_image
from rasterio.io import MemoryFile
from rasterio.plot import show, show_hist
from rasterio.features import shapes
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from typing import Optional


# Load Sentinel-1 VV .tif file

#driver: the name of the desired format driver
#width: the number of columns of the dataset
#height: the number of rows of the dataset
#count: a count of the dataset bands
#dtype: the data type of the dataset
#crs: a coordinate reference system identifier or description
#transform: an affine transformation matrix, and
#nodata: a “nodata” value
def load_and_normalize_image(path):
    with rasterio.open(path) as src:
        img = src.read(1)  # Read first band
        meta = {
        "count": src.count,
        "height": src.height,
        "width": src.width,
        "crs": str(src.crs) if src.crs else None,
        "dtype": src.dtypes[0],
        "nodata": src.nodata,
        "transform": src.transform,
        "driver": src.driver,
        }
    img = np.nan_to_num(img)
    scaler = MinMaxScaler()
    img = scaler.fit_transform(img)
    
    return img, meta

def meta_to_dataframe(meta: dict, name:Optional[str] = None) -> pd.DataFrame:
    """Convert rasterio metadata dictionary to a pandas DataFrame."""
    try:
        res_x = abs(meta["transform"].a)
        res_y = abs(meta["transform"].e)
        px_area = res_x * res_y if np.isfinite(res_x) and np.isfinite(res_y) else np.nan
    except Exception:
        res_x = res_y = px_area = np.nan

    # Bounds del raster (left, bottom, right, top)
    try:
        left, bottom, right, top = array_bounds(
            meta["height"], meta["width"], meta["transform"]
        )
    except Exception:
        left = bottom = right = top = np.nan

    # EPSG (si es convertible)
    try:
        epsg = CRS.from_string(meta["crs"]).to_epsg() if meta["crs"] else None
    except Exception:
        epsg = None

    row = {
        "name": name,
        "driver": meta.get("driver"),
        "dtype": meta.get("dtype"),
        "count": meta.get("count"),
        "height": meta.get("height"),
        "width": meta.get("width"),
        "crs": meta.get("crs"),
        "epsg": epsg,
        "nodata": meta.get("nodata"),
        "res_x": res_x,
        "res_y": res_y,
        "pixel_area": px_area,  # en unidades del CRS (p. ej., m² si CRS está en metros)
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
        "transform": str(meta.get("transform")),  # string para que se muestre bonito en tablas
    }
    return pd.DataFrame([row])


def display_image(img):
    fig, ax = plt.subplots()
    #im = ax.imshow(img, cmap='gray')
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    im = ax.imshow(img, cmap="gray")
    ax.set_title("Imagen Sentinel-1 Normalizada")
    plt.colorbar(im, ax=ax)
    fig1, ax1 = plt.subplots()
            # For single band:
    show_hist(img, bins=50, title="Histogram", ax=ax1)
            # If it’s a 3-band RGB, you can do: show(src, ax=ax1)  # auto RGB
    ax1.set_title("Preview (Band 1)")
    ax1.axis("off")
    st.pyplot(fig1)
    return fig

def process_file():
    st.title("Procesador de Imagenes satelitales")
    # Upload file
    uploaded_file = st.file_uploader("Upload your image to process file", type=["tif","tiff"], accept_multiple_files=True)
    if uploaded_file is not None:
        file_names = [file.name for file in uploaded_file]
        selected_file_name = st.selectbox("Selecciona una imagen para mostrar:", file_names)
        # Get the selected file object
        selected_file = next((file for file in uploaded_file if file.name == selected_file_name), None)
        #selected_file = file for file in uploaded_file if file.name == selected_file_name

        # Guardar archivo temporalmente
        try :
            st.write(f'###Imagen  {selected_file.name}')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                tmp.write(selected_file.getbuffer())
                tmp_path = tmp.name
            # Procesar imagen
            img, meta = load_and_normalize_image(tmp_path)
            meta_df = meta_to_dataframe(meta, name=selected_file.name)
            # Mostrar imagen normalizada
            st.pyplot(display_image(img))
            vals = img[~np.isnan(img)].ravel()
            rng = np.random.default_rng(42)
            sample_size = min(200_000, vals.size) 
            #with img:
            #count, H, W = meta["count"], meta["height"], meta["width"]
            #nodata = meta["nodata"]
            #arr = img
            ##procesar metadata de imagen
            #flat = arr.reshape(-1, 1).T  # (pixels, bands
            sample = vals if vals.size <= sample_size else rng.choice(vals, size=sample_size, replace=False)
            with st.expander("Mostrar metadata de la imagen"):
                st.dataframe(meta_df)
                df = pd.DataFrame({"Band_1": sample})
            #df = pd.DataFrame(flat, columns=["Band_1" ])
                st.dataframe(df)
            # Eliminar archivo temporal
            #os.remove(selected_file)
        except Exception as e:
            st.error(f"Error al procesar {selected_file.name}: {e}")
def proceso_dataset():
    
    def read_tif_as_tensor(path):
        with rasterio.open(os.path) as src:
            img = src.read(1)
            img = np.nan_to_num(img)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # normalización
            img = img.astype(np.float32)
        return tf.convert_to_tensor(img[..., np.newaxis])  # [H, W, 1]
    def read_tif_as_tensor(path):
        path = path.numpy().decode("utf-8")
        with rasterio.open(path) as src:
            img = src.read(1)
            img = np.nan_to_num(img)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # normalización
            img = img.astype(np.float32)
        img = img[..., np.newaxis]  # [H, W, 1]
        img = tf.convert_to_tensor(img)
        img = tf.image.resize(img, [128, 128])
        return img

    def load_image_mask(img_path, mask_path):
        img = tf.py_function(read_tif_as_tensor, [img_path], tf.float32)
        mask = tf.py_function(read_tif_as_tensor, [mask_path], tf.float32)
        img.set_shape([128, 128, 1])
        mask.set_shape([128, 128, 1])


        return img, mask
    def augment(img, mask):
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        return img, mask
    image_paths = sorted([os.path.join("Dataset/Imagen", f) for f in os.listdir("Dataset/Imagen")])
    mask_paths = sorted([os.path.join("Dataset/Agua", f) for f in os.listdir("Dataset/Agua")])

    def build_dataset(img_paths, mask_paths, batch_size=16, shuffle=True):


        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        dataset = dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
    def build_unet(input_shape=(128, 128, 1)):
        inputs = tf.keras.Input(input_shape)

        # Encoder (contracción)
        c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        # Bottleneck
        c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

        # Decoder (expansión)
        u5 = layers.UpSampling2D((2, 2))(c4)
        u5 = layers.concatenate([u5, c3])
        c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
        c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

        u6 = layers.UpSampling2D((2, 2))(c5)
        u6 = layers.concatenate([u6, c2])
        c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c6)

        u7 = layers.UpSampling2D((2, 2))(c6)
        u7 = layers.concatenate([u7, c1])
        c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c7)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

        model = models.Model(inputs, outputs)
        return model
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=10)
    results = []
    @st.cache_data
    def  model_training():
        for run in range(5):
            seed = random.randint(30, 50)
            st.write(f"\n🔁 Entrenamiento {run+1}/10 — seed={seed}")

            train_imgs, val_imgs, train_masks, val_masks = train_test_split(
                image_paths, mask_paths, test_size=0.3, random_state=seed)

            train_ds = build_dataset(train_imgs, train_masks)
            val_ds = build_dataset(val_imgs, val_masks, shuffle=False)

            model = build_unet()
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            history = model.fit(train_ds, validation_data=val_ds, callbacks=[tensorboard_cb],epochs=10, verbose=0)

            acc = history.history['val_accuracy'][-1]
            results.append((seed, acc))
            st.write(f"✅ Accuracy final (validación): {acc:.4f}")
            
            #model_path = f"modelos/model_run{run+1}_seed{seed}.h5"
            
            model_path=model
        return model_path, train_imgs, train_masks
    if st.button("Iniciar Entrenamiento del Modelo"):
        model_path, train_imgs, train_masks = model_training()
        # Cargar modelo de la corrida que deseas probar (ajusta número si es necesario)
    model_path = "modelos/model_run1_seed30.h5"
    if 'key' not in st.session_state:
        st.session_state['key'] = model_path
    st.session_state['train_imgs'] = train_imgs
    st.session_state['train_masks'] = train_masks
    model = load_model(st.session_state['key'])
    st.session_state['model'] = model
    def build_dataset_with_names(img_paths, mask_paths, batch_size=16):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

        def load_with_name(img_path, mask_path):
            img = tf.py_function(read_tif_as_tensor, [img_path], tf.float32)
            mask = tf.py_function(read_tif_as_tensor, [mask_path], tf.float32)
            img.set_shape([128, 128, 1])
            mask.set_shape([128, 128, 1])
            return img, mask, img_path

        dataset = dataset.map(load_with_name, num_parallel_calls=tf.data.AUTOTUNE)
        #dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def visualizar_batch(dataset,n):


    # Extraer un batch
        for imgs, masks, name in dataset.take(1):
            imgs = imgs.numpy()
            masks = masks.numpy()
            name_strings = [p.numpy().decode("utf-8").split("/")[-1] for p in name]
            plt.figure(figsize=(n * 3, 6))
            subplot_titles = [name_strings[i] for i in range(n)] + [f"Máscara {i+1}" for i in range(n)]

            
            for i in range(n):
                # Imagen
                
                plt.subplot(4, n, i + 1)
                plt.imshow(imgs[i].squeeze(), cmap='gray')
                plt.title(f"{name_strings[i]}")
                plt.axis('off')

                # Máscara
                plt.subplot(4, n, i + 1 + n)
                plt.imshow(masks[i].squeeze(), cmap='Blues')
                plt.title(f"Máscara {i+1}")
                plt.axis('off')
                
            plt.tight_layout()
            #st.plotly_chart(fig)
            plt.show()
            st.pyplot(plt)
            break  # solo el primer batch

# Ejemplo de uso
    vis_ds = build_dataset_with_names(st.session_state['train_imgs'], st.session_state['train_masks'])
    visualizar_batch(vis_ds, n=8)
    def read_tif(path):
        with rasterio.open(path) as src:
            img = src.read(1)
            img = np.nan_to_num(img)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32)
    def insumos():
        img_test = "Dataset/Imagen/Chipata.tif"
        img_test_1 = "Dataset/Agua/Chipata_WaterMask.tif"
        test= read_tif(img_test)
        test_1= read_tif(img_test_1)
        sized= tf.image.resize(test[...,np.newaxis], [128, 128])
        sized= tf.expand_dims(sized, axis=0)

        #test_image= read_tif_as_tensor("drive/MyDrive/Dataset/Imagen/Aguada.tif")
        predicted_mask= model.predict(sized)[0,:,:,0]
        predicted_mask_bin = (predicted_mask > 0.1).astype(np.uint8)
        return test, test_1, predicted_mask_bin
    test, test_1, predicted_mask_bin = insumos()
    def comparativo(test, test_1, predicted_mask_bin):
        plt.figure(figsize=(10, 4))

        plt.subplot(2, 2, 1)
        plt.imshow(test, cmap='gray')
        plt.title("Imagen Sentinel-1 Normalizada")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(predicted_mask_bin, cmap='Blues')
        plt.title("Predicción: Cuerpos de Agua")
        plt.axis('off')


        plt.subplot(2, 1, 2)
        plt.imshow(test_1, cmap='Blues')
        plt.title("Detección de agua original")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        st.pyplot(plt)
    st.subheader("Comparativo de Resultados")
    comparativo(test, test_1, predicted_mask_bin)
    LOGDIR = st.text_input("Logdir", value="logs/fit")

    @st.cache_resource
    def start_tensorboard(logdir: str) -> str:
        tb = program.TensorBoard()
        # elige puerto y host; en local usa localhost
        tb.configure(argv=["--logdir", logdir, "--host", "localhost", "--port", "6006"])
        url = tb.launch()  # p.ej. "http://localhost:6006/"
        return url

    if st.button("Launch / Reload TensorBoard"):
        url = start_tensorboard(LOGDIR)
        st.success(f"TensorBoard running at {url}")

    # Si ya está en marcha (por cache), embébelo:
    try:
        url = start_tensorboard(LOGDIR)
        st.components.v1.iframe(url, height=900)
    except Exception as e:
        st.warning(f"TensorBoard not started yet: {e}")
def proceso_post_hoc():
    def load_patch_from_tif(tif_path, patch_size=128):
        with rasterio.open(tif_path) as src:
            img = src.read(1)
            img = np.nan_to_num(img)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        h, w = img.shape
        i = (h - patch_size) // 2
        j = (w - patch_size) // 2
        patch = img[i:i+patch_size, j:j+patch_size]
        return patch
    
    def lime_model_wrapper(images_rgb):
        images = np.array([img[:, :, 0] for img in images_rgb])  # quitar RGB fake
        images = images[..., np.newaxis]  # volver a [N, H, W, 1]
        model=st.session_state['model']
        preds = model.predict(images)
        return np.array([np.mean(p) for p in preds])[:, np.newaxis]

    def segmentation_predict(images_rgb):
        """
        images_rgb: lista de imágenes RGB falsas de shape [H, W, 3]
        Devuelve: array de forma [N, 1] con la predicción media de agua por imagen
        """
        images = np.array([img[:, :, 0] for img in images_rgb])  # usar solo 1 canal
        images = images[..., np.newaxis]  # [N, H, W, 1]
        model=st.session_state['model']
        preds = model.predict(images)
        return np.array([np.mean(p) for p in preds])[:, np.newaxis]
    
    def apply_lime_to_patch(image_patch):
        image_rgb = np.repeat(image_patch[..., np.newaxis], 3, axis=2)  # convertir a RGB

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_rgb.astype('double'),
            segmentation_predict,
            top_labels=1,
            hide_color=0,
            num_samples=100
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, hide_rest=False)

        # Visualización
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image_patch, cmap='gray')
        plt.title("Imagen Sentinel-1")

        plt.subplot(1, 2, 2)
        plt.imshow(mark_boundaries(temp, mask))
        plt.title("LIME - Regiones influyentes")
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)
    
    patch = load_patch_from_tif("Dataset/Imagen/Barrancabermeja.tif", patch_size=128)
    apply_lime_to_patch(patch)
def etapas():
    st.sidebar.title("Explicabilidad XAI")
    option = st.sidebar.selectbox("Selecciona una etapa:", ["Explicabilidad Data", "Explicabilidad Modelo", "Explicabilidad post_hoc"])
    if option == "Explicabilidad Data":
        process_file()
    if option == "Explicabilidad Modelo":
        st.write("Explicabilidad Modelo")
        proceso_dataset()
    if option == "Explicabilidad post_hoc":
        st.title("Explicabilidad post_hoc")
        proceso_post_hoc()
            
if __name__ == "__main__":
    etapas()

