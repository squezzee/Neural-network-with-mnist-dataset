import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
import pathlib
import datetime
import cv2
from tensorflow.keras import layers
from tensorflow.python.data import AUTOTUNE
import sys

# W skrypcie wykorzystałem stronę https://www.tensorflow.org/tutorials/load_data/images
# Co udało się zrobić:
# Stworzyć sieć rozpoznającą dane w databazie litery z dokładnością większą niż 75%
# Zwizualizować proces uczenia za pomocą tensorboard
# Dodano możliwość sprawdzenia sieci na przykładzie zewnętrznego obrazka
# Eksport datasetu do tfrecord
# Eksport sieci do h5 i json

def preprocess_image(img_height, img_width):
  # wpisanie skip pominie opcje sprawdzenia zdjęcia na modelu
  path = input("Podaj lokalizacje zdjęcia: ")
  # gdy podana została ścieżka do pliku to należy dane zdjęcie przystosować do wczytania go do modelu
  if path != "skip":
    image = cv2.imread(path)
    image = cv2.resize(image, dsize=(img_height, img_width), interpolation=cv2.INTER_AREA)
    img_preprocessed = np.expand_dims(image, axis=0)
    return img_preprocessed
  return -1


# funkcje zaczerpnięte z powyższego linka, służą do podzielenia danych na 3 bazy, zwracają parę zdjęcie-podpis
def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  one_hot = parts[-2] == class_names
  return tf.argmax(one_hot)


def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)
  return tf.image.resize(img, [180, 180])


def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


def configure_for_performance(ds, batch_size):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


# zdefiniowanie zmiennych
image_width = 180
image_height = 180
batch_size = 32

# domyślnie baza znajduje się w folderze projektu
data_dir = pathlib.Path("dataset/img")
# obliczam ile zdjęć znajduje się w bazie (aby móc sensownie podzielić bazę na 3 mniejsze) i tasuje dane
image_count = len(list(data_dir.glob('*/*')))
list_ds = tf.data.Dataset.list_files('dataset/img/*/*', shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=True)

# tworzę słownik aby móc łatwo rozszyfrować nazwy folderów jako konkretne litery (zakładam że w folderze nie ma innych
# plików/folderów, w przeciwnym razie należy daną nazwę wpisać w miejsce "przyklad"
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "przyklad"]))
letters = ['M', 'O', 'R', 'S', 'W', 'B', 'C', 'F', 'I', 'P']
letters_dict = dict(zip(class_names, letters))

# wyznaczam rozmiary poszczególnych baz
train = int(0.8*image_count)
val = int(0.1*image_count)
test = int(0.1*image_count)

# przydzielam potasowane dane do odpowiednich baz
train_dataset = list_ds.take(train)
test_dataset = list_ds.skip(train)
val_dataset = list_ds.skip(test)
test_dataset = list_ds.take(test)

# wywołanie num_parallel_calls powoduje zrównoleglenie wczytywania i obróbki zdjęć
train_dataset = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(process_path, num_parallel_calls=AUTOTUNE)

# normalizacja danych przed wczytaniem ich do sieci
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

# przyśpieszenie procesu z użyciem cache()
train_dataset = configure_for_performance(train_dataset, batch_size)
val_dataset = configure_for_performance(val_dataset, batch_size)
test_dataset = configure_for_performance(test_dataset, batch_size)

# stworzenie warstw modelu oraz jego kompilacja
model = tf.keras.Sequential([
  preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(10, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# użycie tensorboard oraz dodanie callbacku
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# uczenie modelu
model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=[tensorboard_callback])

# ewaluacja modelu na zbiorze testowym
loss_evaluate, accuracy_evaluate = model.evaluate(test_dataset, verbose=2)

# zapisanie modelu do rozszerzenia json, wag do pliku .h5
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

# zapisanie datasetu do formatu .tfrecord
writer = tf.data.experimental.TFRecordWriter("dataset.tfrecord")
writer.write(list_ds)

# wczytanie obrazu wybranego przez użytkownika do przetestowania działania sieci
img_batch = preprocess_image(image_height, image_width)

if not isinstance(img_batch, int):
  predictions = model.predict(img_batch)
  index = np.argmax(predictions)
  print("Ten obrazek to " + str(letters_dict[class_names[index]]) + " z prawdopodobieństwem " + str(predictions[0, index]))

# automatyczne wywołanie tensorbordu, wystarczy wejść w link wyświetlony w konsoli
sys.stdout.write("Automatycznie wywołano tensorboard \n ")
os.system("tensorboard --logdir logs/fit")



