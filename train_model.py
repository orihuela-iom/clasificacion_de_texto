"""
Modulo para la selección del modelo 
Author: Ismael Orihuela
"""
from clean.clean_text import TicketMessage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
import numpy as np
import math
from joblib import load
from lime.lime_text import LimeTextExplainer


class SelectModel():
    '''
    Compara varios modelos con configuraciones por default
    usando validacion cruzada estratificada
    con el objetivo de selecionar el mejor estimador.

    Parameters:
    -----
        x: variable predictoras
        y: datos respuesta
    '''
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.evaluation: dict = {}
        self.summary_scores: dict = {}
        self.random_seed:int = 2023

        self.default_models = {
            "multinomial_lg": LogisticRegression(multi_class='multinomial',
                                                 solver='lbfgs', max_iter=500),
            "niaveB": ComplementNB(),
            "extra_tree": ExtraTreesClassifier(),
            "random_forest": RandomForestClassifier(),
            "knn_class": KNeighborsClassifier()
            }

        self.cv_method = StratifiedKFold(n_splits=3)


    def run_cv(self, model_name:str):
        '''
        Realiza la evaluacion de un modelo medinte SKCV y guarda la evalucion del modelo

        Parameters: 
        ----
            model_name (str): nombre del modelo a evaluar
                              solo aplicada para los modelos asignados en default_models
        '''

        cv_scores = cross_validate(
            self.default_models[model_name],
            self.x, self.y,
            scoring=['accuracy', 'f1_macro',
                    'precision_macro', 'recall_macro'], 
                    cv=self.cv_method,
            return_estimator=True,
            n_jobs=1)

        # mejor estimador
        cv_scores["estimator"] = cv_scores["estimator"][np.argmax(cv_scores["test_f1_macro"])]

        # guardar resultados
        self.evaluation[model_name] = cv_scores


    def evaluate_models(self):
        '''
        Itera sobre default_models para realizar la evalaucion
        '''
        for model in self.default_models:
            self.run_cv(model)
            self.summary()


    def summary(self):
        '''
        Calcula el promedio de las metricas evaluadas para cada modelo
        y lo guarda en un dataframe
        '''
        for model_scores in self.evaluation.keys():
            temp_summary = {name: scores.mean() for name, scores in
                            self.evaluation[model_scores].items() if name.startswith("test_")}
            self.summary_scores[model_scores] = temp_summary

        self.results = pd.DataFrame.from_dict(self.summary_scores, orient="index")



multi_lr_model = load('model/mlr.joblib')
explainer = LimeTextExplainer(class_names=list(multi_lr_model.classes_))


def make_predit(raw_text:str) -> dict:
    """
    Realiza predicciones sobre un texto

    Parameters:
    -----
        raw_text(str): Texto en formato libre
    
    Returns:
    -----
        dict: Diccionario con valor predicho por el modolo, texto ingresado, top 5 palabras más 
        relvevantes y probabilades para cada categoría
    """

    if raw_text == "" or raw_text is None or len(raw_text.split()) < 3:
        results = {"Predicted category": "INGRESA UN TEXTO VALIDO",
                          "Input text": "",
                          "Key words": {},
                          "proba": {}
                          }
        return results


    cleaned_text = TicketMessage.parse_text(raw_text)
    prediction = multi_lr_model.predict([cleaned_text])[0]

    results = {"Predicted category": prediction}
    results["Input text"] =  raw_text
    exp = explainer.explain_instance(cleaned_text,
                                multi_lr_model.predict_proba,
                                num_features=5,
                                top_labels=1)
    key_words = exp.as_list(label=exp.available_labels()[0])
    results["Key words"] = {value[0]: value[1] for value in key_words}
    results["proba"] = {cat: math.ceil(p*10_000) / 10_000 for cat, p in  zip(exp.class_names, exp.predict_proba)}

    return results
