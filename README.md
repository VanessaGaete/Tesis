# Tesis
Profesor Guía: Andrés Abeliuk
Profesor Co-guía: Naim Bro

Códigos que estoy utilizando para mi investigación en la tesis, focalizada en investigar la utilización de LLM para las ciencias políticas. El trabajo de referencia es el siguiente:

En el caso de este proyecto, se utilizan datos de encuentas públicas para investigar si algún modelo logra entender los comportamientos y opiniones de los chilenos.

Los modelos utilizados en la investigación son: ChatGPT-3.5-turbo, ChatGPT-4, Llama-7b-chat-hf, Llama-13b-chat-hf y Mistral.

Aquí presento solo algunos de los códigos más importantes.
## Elec_pres_querys.py
En este archivo se pueden ejecutar distintos modelos en distintas variantes, y también con distintas variantes de prompts. Se crea un objeto con todos los parámetros corrspondientes que se envían luego a requester_script.py. Algunas de las variables son: Llama7b, Llama13b, Mistral; modelo en inglés o español; con o sin finetuning; con Zero-Shot Learning o Fewshot Learning, entre otras.

## requester_script.py
Este es el archivo que maneja toda la lógica para hacer múltiples consultas al modelo requerido. También se diseña el prompt en el formato necesario. 

## fine-tuning.py
Contiene el código necesario para finetunear ya sea Llama13b o 7b con los datos manejados en la tesis. 

## RAG.ipynb
Estoy comenzando a investigar cómo aplicar Retrieval Augmented Generation a alguno de los modelos para mejorar rendimientos. 
