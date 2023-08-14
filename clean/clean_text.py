"""
Modulo auxiliar en la limpieza de los textos
Author: Ismael Orihuela
"""

import re
import spacy 

# usar el lemmatizador de spacy
nlp = spacy.load('en_core_web_sm',  disable=["parser", "ner"])
stop_words = nlp.Defaults.stop_words


class TicketMessage():
    '''
    Clase auxiliar en la limpiza de texto
    Almacena el valor original del texto
    junto con una version limpia del mismo
    '''
    def __init__(self, raw_text: str) -> None:

        self.raw_text = raw_text
        self.cleaned_text: str = ''
        self.set_clean_word()


    @staticmethod
    def remove_stop_words(text:str) -> str:
        """
        Función intermedia que convetirte a minúscula, quita caracteres no validos y stopwords

        Parameters:
        -----
            text(str): texto sin procesar
        
        Returns:
            str: texo en minúscula, sin stopwords, ni caracteres no validos
        """
        text = text.lower()
        # quitar puntuacion
        text = re.sub(r"[!#$%&()*+,-.:;<=>?@\[\]^_`|~]", "", text)
        text = [word for word in text.split() if word not in stop_words]
        return ' '.join(text)


    @staticmethod
    def remove_other_characters(text: str)  -> str:
        """
        Función intermedia que
        Quita fechas con formato xx/xx/xxxx
        Quita texto con formato xxx
        Quita cantidades con formato {$xx.xx}
        Quita caracteres que no sean letras
        Quita espacios en blanco extras
        Quita números
        """
        # quitar fechas
        text = re.sub(r"\w+\/\w+\/\w+", "", text)

        #Quitar texto con formato xxxx
        text = re.sub(r"x{2,}", "", text)

        # remplazar cantidades {$xx.xx}
        text = re.sub(r"\{[^}]*\}", "", text)

        # quitar caracteres que no sean letras
        text = re.sub(r"[^\w\s]", "", text)

        # quitar guines bajos
        text = re.sub(r"_+", " ", text)

        # quitar numeros
        text = re.sub(r"\d+", "", text)

        # quitar espacion en blanco extra
        text = re.sub(r"\s{2,}", " ", text)

        return text

    @staticmethod
    def lemmatize(text:str) -> str:
        "Función intermedia que convertir a palabras a su forma base"
        # proceso lento
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc])


    @staticmethod
    def parse_text(raw_string: str) -> str:
        '''
        Devuelve texto limpio

        Parameters:
        ----
            raw_string(str): Texto sin procesar

        Returns:
        ----
            str: texto limpio
        '''
        cleaned = TicketMessage.remove_stop_words(raw_string)
        cleaned = TicketMessage.remove_other_characters(cleaned)
        cleaned = TicketMessage.lemmatize(cleaned)

        return cleaned


    def set_clean_word(self):
        """
        Guarda el texto limpio dentro de la clase
        """
        self.cleaned_text = TicketMessage.parse_text(self.raw_text)



if __name__ == "__main__":

    #probar clase
    words = TicketMessage("Hello and bye")
    print(words.cleaned_text)
