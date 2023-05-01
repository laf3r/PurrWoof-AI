import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

#Модель нейронной сети PurrWoof-AI.
#Автор И. А. Герасимов @laf3r | github.com/laf3r

# Константы
EPOCHS = 20
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Загрузка и разделение данных на обучающее и проверочное множества
train_dataset, validation_dataset = tfds.load(
    name='cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    shuffle_files=True #Перемешивание файлов
)
# ограничение в 50 картинок. Для отладки.
# train_dataset = train_dataset.take(50)#берём 50 картинок из тренировочного набора данных
# validation_dataset = validation_dataset.take(50)#берём 50 картинок из проверочного набора данных

# Обработка данных
def preprocess(image, label):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=2)
    return image, label

# Ставим метки
train_dataset = train_dataset.map(preprocess).batch(BATCH_SIZE)
validation_dataset = validation_dataset.map(preprocess).batch(BATCH_SIZE)

# Имена классов
class_names = ["Кошка", "Собака"]

# Определение параметров обучения
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()
metrics = ['accuracy']

# Инициализация модели
# CNN МОДЕЛЬ
# CNN - Свёрточная нейронная сеть
# Для бинарной классификации используется оптимизатор Adam Adaptive Moment Estimation
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

#На выходе два нейрона потому что бинарная классификация, т.е собака и кот - два варианта.

#Компиляция модели
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)
# Обучение модели
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)

# Построение графика точности и потерь
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


test_dataset = validation_dataset.take(10)
test_images, test_labels = next(iter(test_dataset.take(10)))

# Получение предсказаний нейросети для 10 изображений
predictions = model.predict(test_images)

# Преобразование меток из one-hot encoding в обычный вид
test_labels = np.argmax(test_labels, axis=1)

# Вывод 10 изображений и соответствующих им меток и предсказаний
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    # Отображение изображения
    ax.imshow(test_images[i])
    # Отображение меток и предсказаний
    true_label = class_names[test_labels[i]]
    pred_label = class_names[np.argmax(predictions[i])]
    if true_label == pred_label:
        ax.set_title("Это: {}, ИИ: {}".format(true_label, pred_label), color='green')
    else:
        ax.set_title("Это: {}, ИИ: {}".format(true_label, pred_label), color='red')

plt.tight_layout()
plt.show()
