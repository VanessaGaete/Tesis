import pandas as pd
#from requester_v import Exp_Request
from requester_script import Exp_Request
from index_matrix import index_matrix
import dictionaries


df1 = pd.read_csv('exp2_inputs_filtrado.csv')
#df1 = df1.drop(df1.index[:298])
#data = pd.read_csv('exp1_inputs.csv')
fewshot_df = pd.read_csv('./prompt-dataframes/prompt_dataset_presidencial_fewshot.csv')
fewshotexamples_df = pd.read_csv('./prompt-dataframes/prompt_dataset_presidencial_fewshotexamples.csv')


# Definir la clase derivada (clase hija) que hereda de Animal
class Elecpres_Requester(Exp_Request):
  #model,
  def __init__(self, model, b, name, sys=False, eng=False, red=False, wpv=False,with_context = True, spec = False,translate= False, ablation = False, fewshot = False, finetuned=False, random_fewshot =False, nsamples = 0):
    # Llama al constructor de la clase padre usando super()
    #model_llama
    if random_fewshot:
      examples_df = fewshotexamples_df
    else: 
      examples_df = []

    dictionary = dictionaries.voto
    imatrix = index_matrix
    jsonl_name = './' + model + '_elecpres_' + b + 'b' +name + '.jsonl'
    if finetuned:
      jsonl_name = './elecpres' +name + '.jsonl'
    #jsonl_name = './respuestas_elecpres_' + b + 'b' +name + '.jsonl'
    
    #####################   Prompts   ######################
    #if finetuned:
    #  task= "Analiza el perfil de la siguiente persona y entrega la respuesta que probablemente daría dicha persona a la pregunta que se hará.\n "
      #inst = ""
    #  end= ""
    if not eng:
      #task="Tu trabajo es ser un asistente que ayudará a predecir el voto de ciertas personas en la segunda vuelta presidencial de Chile del 2021. Debes responder siempre con tan solo una palabra de entre las siguientes opciones: Boric, Kast o Nulo.\n "
      task = "Eres un modelo predictor. Tu tarea es predecir las respuestas de una persona a cierta pregunta basándote en sus características sociodemográficas. Las características sociodemográficas estarán delimitadas con triple backsticks, la pregunta estará delimitada con los siguientes símbolos < >, y las alternativas de respuesta por [ ]."
      context = "Primero te voy a dar un poco de contexto en el que debes situarte. La elección presidencial de Chile para el período 2022-2026 se realizó el 21 de noviembre de 2021, la segunda vuelta electoral, en tanto, tuvo lugar el 19 de diciembre de 2021.\n El resultado de la primera vuelta presidencial ubica a José Antonio Kast en primer lugar con un 27,91% de los votos, a Gabriel Boric Font en segundo lugar con un 25.83% de los votos, el resto de los candidatos no lograron obtener más de un 12% de los votos. Por lo tanto, Gabriel Boric y José Antonio Kast fueron los candidatos que pasaron a la segunda vuelta.\n Así los chilenos debían elegir entre un candidato de izquierda y otro de derecha respectivamente. El enfoque principal de la candidatura de Gabriel Boric era igualdad de género, seguridad social y sustentabilidad. Por otro lado el candidato José Antonio Kast realizó una campaña centrada en seguridad, orden, disminución del Estado e instalándose como opositor a los sucesos relacionados con el estallido social del 2019.\n"
      end_JSON = "Genera solo un JSON (sin palabras ni explicaciones extras) con las siguientes llaves: 'prediccion' que muestre solo la opción elegida de entre las alternativas que se te darán, y 'explicacion' (en máximo 100 palabras) conteniendo la razón por la que elegiste dicha alternativa."
      end = "A continuación se presenta una pregunta con las opciones de respuesta. Entrega un JSON con las siguientes llaves: 'prediccion' conteniendo la alternativa elegida para responder a la pregunta, 'explicación' (en no más de 100 palabras) explicando por qué elegiste esa alternativa."
    else:
      task =  "Your job is to be an assistant who will help predict the vote of certain people in the 2021 Chilean presidential second round. You must always respond with just one word from the following options: Boric, Kast or Nulo.\n"
      context =  "First I'm going to give you a little context in which you should place yourself. The presidential election of Chile for the period 2022-2026 was held on November 21, 2021, the second electoral round, meanwhile, took place on December 19, 2021.\n The result of the first presidential round places José Antonio Kast in first place with 27.91% of the votes, Gabriel Boric Font in second place with 25.83% of the votes, the rest of the candidates failed to obtain more than 12% of the votes. Therefore, Gabriel Boric and José Antonio Kast were the candidates who went to the second round.\n Thus, Chileans had to choose between a left-wing candidate and a right-wing candidate respectively. The main focus of Gabriel Boric's candidacy was gender equality, social security and sustainability. On the other hand, the candidate José Antonio Kast carried out a campaign focused on security, order, reduction of the State and establishing himself as an opponent of the events related to the social outbreak of 2019.\n"
      end = "Now analyze the profile of the following person and answer the question that will be asked of you. \n"

    if spec:
      inst = " Por favor también responde en a lo más 200 palabras por qué crees que la persona eligiría esa opción y en qué variables te fijaste al hacer la predicción. Usa la primera palabra para indicar la opción elegida y el resto de palabras para explicar el por qué."
    else:  
      inst = "Recuerda responder en solo 1 palabra. \n"

    if not with_context:
      context = ""

    super().__init__(1, model, b, jsonl_name,  task, context, end , inst, dictionary, sys,eng, red, wpv,translate, ablation, fewshot, finetuned, nsamples, random_fewshot, examples_df, imatrix )
    #super().__init__(1, model_name, jsonl_name,  task+context, end , inst, dictionary, eng, red, wpv, ablation, fewshot, nsamples, random_fewshot, examples_df, imatrix )
              

########################################### Función Requests ###################################################
b = str(7)

################################################### Original ###################################################
#r = Elecpres_Requester("mistral", "v0.2", "_sys", sys=True)
#r.answers_generator(df1)

#r=Elecpres_Requester("llama", str(7), "_final")
#r.answers_generator(df1)

#r=Elecpres_Requester("llama", str(13), "_final")
#r.answers_generator(df1)

#r=Elecpres_Requester("llama-2-13b-datacamp_v5_PRES_ENG",str(13), "A", finetuned = True)
r=Elecpres_Requester("llama-2-13b-datacamp_v6_ENG_ONLYPRES",str(13), "13b_v6_ONLYPRES_ENG.jsonl", translate=True, finetuned = True)
r.answers_generator(df1)

#r = Elecpres_Requester("mistral", "v0.2", "_final")
#r.answers_generator(df1)

################################################## RANDOM FEWSHOT ###################################################
#r = Elecpres_Requester(b, "_rfewshot", random_fewshot =True, examples_df = fewshotexamples_df)
#r.answers_generator(fewshot_df) #3, 6

################################################### SPEC ###################################################
#answers_generator(dataset, b, task, context, end, spec, name = "_spec")

###################################################  WPV  ###################################################
#answers_generator(dataset_wpv, b, task, context, end, inst, name = "_wpv")

""" r = Elecpres_Requester("mistral", "v0.2", "_sys_wpv", sys=True)
r.answers_generator(df1)

r = Elecpres_Requester("mistral", "v0.2", "_wpv")
r.answers_generator(df1)
 """

###################################################  REDUCED  ###################################################
""" #answers_generator(dataset_red, b, task, context, end, inst, name = "_red")
r = Elecpres_Requester("mistral", "v0.2", "_sys_red", sys=True)
r.answers_generator(df1)

r = Elecpres_Requester("mistral", "v0.2", "_red")
r.answers_generator(df1) """
###################################################  ENG  ###################################################
#answers_generator(dataset_eng, b, task_eng, context_eng, end_eng, "", name = "_eng", random_fewshot=True, nsamples = 1)
#r = Elecpres_Requester("mistral", "v0.2", "_sys_red", sys=True, with_context = False )
#r.answers_generator(df1)

# r=Elecpres_Requester("respuestas", str(7), "_eng", eng =True)
# r.answers_generator(df1)

# r=Elecpres_Requester("respuestas", str(13), "_eng", eng =True)
# r.answers_generator(df1)

#r = Elecpres_Requester("mistral", "v0.2", "_eng_resto", eng= True)
#r.answers_generator(df1)
############################################## WITHOUT CONTEXT ####################################
# r = Elecpres_Requester("llama", b, "_withoutcontext", with_context = False)
# r.answers_generator(df1) #3, 6 

#r = Elecpres_Requester("mistral", "v0.2", "_sys_withoutcontext", sys=True, with_context = False )
#r.answers_generator(df1)

#r = Elecpres_Requester("mistral", "v0.2", "_withoutcontext", with_context = False)
#r.answers_generator(df1) 

###################################################  ABLATION  ###################################################
#r = Elecpres_Requester(str(7), "_ablation", ablation = True)
#r.ablation_request(df1) #3, 6

#r = Elecpres_Requester(str(13), "_ablation", ablation = True)
#r.ablation_request(df1) #3, 6
