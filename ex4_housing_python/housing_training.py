import os

try:
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import layers, models # type: ignore
except ImportError:
  print("Trying to install: pandas")
  os.system('python -m pip install pandas')
  print("Trying to install: numpy")
  os.system('python -m pip install numpy')
  print("Trying to install: tensorflow")
  os.system('python -m pip install tensorflow')
  print("Trying to install: scikit-learn")
  os.system('python -m pip install scikit-learn')
  os.system('conda update conda')
# -- above lines try to install requests module if not present
# -- if all went well, import required module again ( for global access)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models # type: ignore



# Importazione del dataset
dataset_url = "https://raw.githubusercontent.com/timothypesi/Data-Sets-For-Machine-Learning-/main/california_housing_train.csv"
df = pd.read_csv(dataset_url)

# # Rimozione delle colonne non necessarie
# df = df.drop(columns=["median_income", "households"])

# Suddivisione del dataset in training set e test set
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Preparazione dei dati per il modello TensorFlow
train_labels = train_df.pop('median_house_value')
test_labels = test_df.pop('median_house_value')

# Normalizzazione dei dati
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_df))

# Costruzione del modello ANN con TensorFlow
model = models.Sequential([
    normalizer,
    layers.Dense(200, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Addestramento del modello
model.fit(train_df, train_labels, epochs=100, validation_split=0.2)

# Valutazione del modello sul test set
loss = model.evaluate(test_df, test_labels)

# Predizione dei valori sul test set
test_predictions = model.predict(test_df).flatten()

# Aggiunta delle predizioni al test set
test_df['predicted_median_house_value'] = test_predictions

# Salvataggio del modello e dei risultati
model.save("Users/g.scorpaniti/notebook/ex4_housing_python/artifacts/model")
test_df.to_csv("Users/g.scorpaniti/notebook/ex4_housing_python/artifacts/test_set.csv", index=False)
