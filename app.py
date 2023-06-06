import matplotlib as mpl
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import models
import io
import streamlit as st
from PIL import Image, ImageFilter


def load_img(path_to_img):
    max_dim = 512
    img = path_to_img
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)
    img = tf.keras.utils.array_to_img(img)

    # Нам нужно транслировать массив изображений так, чтобы он имел пакетное измерение
    img = np.expand_dims(img, axis=0)
    return img


def img(uploaded_file_1, uploaded_file_2):
    if st.button('СТАРТ'):
        if uploaded_file_1 is not None and uploaded_file_2 is not None:
            # Получение загруженного изображения
            image_data_1 = uploaded_file_1.getvalue()
            image_data_2 = uploaded_file_2.getvalue()
            content_path = Image.open(io.BytesIO(image_data_1))
            content = load_img(content_path)
            style_path = Image.open(io.BytesIO(image_data_2))
            style = load_img(style_path)
            return content_path, style_path, content, style


def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Вход для депроцессирования изображения должен быть изображением "
                               "размер [1, высота, ширина, канал] или [высота, ширина, канал]")
    if len(x.shape) != 3:
        raise ValueError("Неверный ввод для обратной обработки изображения")

    # Выполняем обратный шаг предварительной обработки
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Слой контента для сброра карты признаков
content_layers = ['block5_conv2']

# Слои стилей которые нам нужны
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def get_model():
    # Загружаем нашу модель. Загружаем предварительно обученный VGG, обученный на данных imagenet
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Получение выходных слоев, соответствующих слоям стиля и содержимого
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Построение модели
    return models.Model(vgg.input, model_outputs)


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    # Сначала мы создаем каналы изображения
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    # Масштабируем потери на данном слое по размеру карты объектов и количеству фильтров
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)


def get_feature_representations(model, content_path, style_path):
    # Загружаем изображения
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # Вычисление признаков стиля и контента
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Получение представления признаков
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    # Пропускаем наше инициализированное изображение через модель
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Собираем все потери по стилю
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Собираем все потери по контенту
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Итоговые потери
    loss = style_score + content_score
    return loss, style_score, content_score


def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Вычисляем градиент относительного одного изображения
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    model = get_model()
    # Нам не нужно обучать слои, поэтому сделаем их False
    for layer in model.layers:
        layer.trainable = False

    # Получаем представления элементов стиля и содержимого (из указанных нами промежуточных слоев)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Устанавливаем начальное изображение
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Создаем оптимизатор
    opt = tf.optimizers.Adam(learning_rate=5, epsilon=1e-1)

    # Сохраняем лучший результат
    best_loss, best_img = float('inf'), None

    # Создаем конфигурацию
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    progress_text = "Идет обработка, пожалуйста, подождите :)"
    my_bar = st.progress(0, text=progress_text)
    for i in range(1, num_iterations + 1):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        print('Iteration: {}'.format(i))
        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
        my_bar.progress((i * 100) // num_iterations, text=progress_text)

    return best_img, best_loss

try:

    st.title("Давайте начнем!")
    uploaded_file_1 = st.file_uploader(label='Выберите первое изображение', type=["jpg", "png"])
    uploaded_file_2 = st.file_uploader(label='Выберите второе изображение', type=["jpg", "png"])
    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file_1:
            st.subheader("Картинка")
        st.image(uploaded_file_1)


    with col2:
        if uploaded_file_2:
            st.subheader("Стиль")
        st.image(uploaded_file_2)


    list_img = img(uploaded_file_1, uploaded_file_2)

    content_path, style_path, content, style = list_img
    best, best_loss = run_style_transfer(content_path,
                                         style_path, num_iterations=50)
    st.balloons()
    st.title("Итог")
    # st.image(best)
    final_img = Image.fromarray(best)
    final_img.save('img.png')
    final_img = final_img.filter(ImageFilter.SHARPEN)
    image_data_1 = uploaded_file_1.getvalue()
    content_path = Image.open(io.BytesIO(image_data_1))
    final_img = final_img.resize((content_path.size[0],content_path.size[1]))

    # if key("S", ):
    #     with open('img.png', "rb") as file:
    #         btn = st.download_button(
    #             label="Скачать",
    #             data=file,
    #             file_name="my_img.png",
    #             mime="image/png"
    #         )
    st.image(final_img,channels="BGR", output_format="PNG")

except Exception:
    pass
