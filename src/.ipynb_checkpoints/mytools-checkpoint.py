import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing import image
from time import time


# La fonction **log_mlflow_run** ci-dessous permet de standariser le tracking des données via `MLflow`. Elle permet de sauvegarder les modèles entraînés ainsi que leurs principales caractéristiques : *paramètres*, *métriques*, *tags* et éventuels *artifacts*.

def log_mlflow_run(run_name, parameters, metrics, model=None, model_name=None, model_type='sklearn', tags=None, artifacts=None):
    """
    Fonction pour logger les paramètres, métriques, tags et artifacts dans MLflow.
    
    Arguments:
    - parameters : Dictionnaire des paramètres du run.
    - metrics : Dictionnaire des métriques à suivre.
    - model : Modèle à enregistrer.
    - model_name : nom du modèle à enregistrer.
    - tags : Dictionnaire des tags à ajouter à l'expérience.
    - artifacts : Dictionnaire des chemins des artifacts à ajouter à l'expérience.

    """
    # Vérifier s'il y a un run actif du même nom
    if mlflow.active_run():
        mlflow.end_run()
    
    # Démarre un nouveau run
    mlflow.start_run(run_name=run_name)
    
    # Ajout des paramètres
    for key, value in parameters.items():
        mlflow.log_param(key, value)
        
    # Ajout des métriques
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    
    try:
        # Ajout des tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        # Ajout des artifacts
        if artifacts:
            for key, value in artifacts.items():
                mlflow.log_artifact(value, key)

        # Enregistrement du modèle
        if model:
            if model_type == 'sklearn':
                mlflow.sklearn.log_model(model, model_name)
            elif model_type == 'tensorflow':
                mlflow.tensorflow.log_model(model, model_name)
            elif model_type == 'keras':
                mlflow.keras.log_model(model, model_name)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")
              
    except Exception as e:
        print(f"Erreur lors de la journalisation dans MLflow: {e}")
        
    finally:
        # Fin du run
        mlflow.end_run()


def compile_and_train(model,
                      train_data,
                      val_data,
                      loss='categorical_crossentropy',
                      optimizer=Adam,
                      learning_rate=0.001,
                      metrics=['accuracy', 'precision', 'recall', 'auc'],
                      epochs=20,
                      stopping=10,
                      lr_patience=5,
                      factor=0.5,
                      verbose=1
                     ):
    
    # Compilation du modèle
    model.compile(loss=loss,
                  optimizer=optimizer(learning_rate=learning_rate),
                  metrics=metrics
                 )
    
    # Construction des callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=stopping, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=lr_patience, min_lr=0.0001)
    
    debut = time()
    
    # Entraînement du modèle
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=epochs,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=verbose
                       )
    fin = time()
    
    duree = fin - debut
    
    return history, round(duree/60)



def graphique_auc_perte(history, model_name):
    
    # Récupérer les données d'entraînement
    auc = history.history['auc']
    val_auc = history.history['val_auc']
    
    # Récupérer les données de perte
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    fig, axe = plt.subplots(1,2, figsize = (10,5))
    axe = axe.flatten()

    # Premier subplot : Courbes des AUCs
    axe[0].plot(epochs, auc, 'b', label='Training AUC')
    axe[0].plot(epochs, val_auc, 'r', label='Validation AUC')
    axe[0].set_title('Training and validation AUC')
    axe[0].set_xlabel('Epochs')
    axe[0].set_ylabel('AUCs')
    axe[0].legend()
    axe[0].grid(True)

    # Deuxième subplot : Loss et Validation loss
    axe[1].plot(epochs, loss, 'b', label='Training loss')
    axe[1].plot(epochs, val_loss, 'r', label='Validation loss')
    axe[1].set_title('Training and validation loss')
    axe[1].set_xlabel('Epochs')
    axe[1].set_ylabel('Loss')
    axe[1].legend()
    axe[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'../outputs/courbes_performances_{model_name}.png')
    plt.show()


def matrice_confusion(model, model_name, val_data, class_names=None):
    """
    Affiche la matrice de confusion pour un modèle donné et des données de validation.

    Parameters:
    model : tensorflow.keras.Model
        Le modèle entraîné.
    val_data : tensorflow.keras.preprocessing.image.DirectoryIterator
        Les données de validation.
    class_names : list, optional
        Les noms des classes. Par défaut, None.
    """
    
    # Prédiction des étiquettes sur les données de validation
    val_preds = model.predict(val_data)
    val_preds_classes = np.argmax(val_preds, axis=1)
    
    # Récupération des vraies étiquettes
    val_true_classes = val_data.classes

    # Calcul de la matrice de confusion
    cm = confusion_matrix(val_true_classes, val_preds_classes)
    
    # Utilisation de ConfusionMatrixDisplay pour afficher la matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Matrice de confusion')
    plt.savefig(f'../outputs/matrice_confusion_{model_name}.png')
    plt.show()

# Fonction pour prédire la race du chien
def prediction_race(model, img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    race = np.argmax(predictions, axis=1)
    return race