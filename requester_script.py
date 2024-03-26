import os
import json
from pprint import pprint
import random

import bitsandbytes as bnb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import aux_ablation
import dictionaries
from prompt_generator import prompt_generator, prompt_generator_english
from deep_translator import GoogleTranslator

import transformers
from datasets import load_dataset
from peft import (
  LoraConfig,
  PeftConfig,
  PeftModel,
  get_peft_model,
  prepare_model_for_kbit_training,
)
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
)


bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.float16,
)

# Definir la clase base (clase padre)
class Exp_Request:

  def __init__(self, exp, model_name, b,jsonl_name, task, context, end , inst, dictionary,sys, eng, red, wpv,translate,ablation, fewshot, finetuned, nsamples, random_fewshot, examples_df, index_matrix):
    self.model_n = model_name
    if finetuned:
      self.model_name = model_name
    elif self.model_n == "mistral" and b == "v0.1":
      self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif self.model_n == "mistral" and b == "v0.2":
      self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    else:
      self.model_name = "meta-llama/Llama-2-" + b + "b-chat-hf"

    print(self.model_name)
    self.finetuned = finetuned
    self.model, self.tokenizer, self.generation_config = self.get_model_params(self.model_name)

    self.translate = translate
    if translate:
      self.translator = GoogleTranslator(source='es', target='en')
      self.task = self.translator.translate(task)
      self.context = self.translator.translate(context)
      self.end = self.translator.translate(end)
      self.inst = self.translator.translate(inst)
    else:
      self.task = task
      self.context = context
      self.end = end
      self.inst = inst

    self.jsonl_name = jsonl_name
    self.fewshot = fewshot
    self.random_fewshot = random_fewshot
    self.examples_df = examples_df
    self.imatrix = index_matrix
    self.nsamples = nsamples
    self.dictionary = dictionary
    self.eng = eng
    self.exp = exp
    self.wpv = wpv
    self.red = red
    self.ablation = ablation
    self.var_removed = ""


    # Load the pre-trained tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    #tokenizer.pad_token = tokenizer.eos_token

  def load_model(self,model_name):

    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      #device_map={"":0},
      device_map="auto",
      quantization_config=bnb_config,
      trust_remote_code=True
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    generation_config = model.generation_config
    generation_config.max_new_tokens = 500
    generation_config.temperature = 1.5
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    #generation_config.repetition_penalty= 0
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer, generation_config
  
  def load_model_finetuned(self,model_name):
    PEFT_MODEL = model_name
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
    )
    config = PeftConfig.from_pretrained(PEFT_MODEL)
    
    # Load the model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
      config.base_model_name_or_path,
      return_dict=True,
      #device_map={"":0},
      device_map = "auto",
      quantization_config=bnb_config,
      trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, PEFT_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    generation_config = model.generation_config
    generation_config.max_new_tokens = 500
    generation_config.temperature = 0.1
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    #generation_config.repetition_penalty=0
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    print(generation_config.max_new_tokens)
    return model, tokenizer, generation_config


  def get_model_params(self, model_name):

    if self.finetuned:
      return self.load_model_finetuned(model_name)
    else:
      return self.load_model(model_name)
    

  def ablation_request(self, dataset):
    #"gender", "age", "region", "indigenous", "gse", "scholarity", "religion",
    variables = [ "gender", "age", "region", "indigenous", "gse", "scholarity", "religion","ideology", "party", "interest"]

    for var in variables:
      self.var_removed = var
      data_ablation = aux_ablation.ablation_df(dataset, var, self.exp)
      print(data_ablation.iloc[0]["prompt"])
      self.answers_generator(data_ablation)

  def answers_generator(self, dataset):
    i=0
    fewshot_text = ""
    if self.fewshot:
      random_indexes = random.sample(range(len(self.examples_df)), self.nsamples)
      #examples_list, answers_list = self.fewshot_examples_generator(dataset, random_indexes)
      prompts_list = list(self.examples_df.iloc[random_indexes]["prompt"])
      answers_list = list(self.examples_df.iloc[random_indexes]["output"])
      dataset = dataset.drop(random_indexes)
      fewshot_text = self.fewshot_prompt_generator(prompts_list, answers_list)
    
    for index, example in dataset.iterrows():
      
      prompt = self.prompting(example, index, fewshot_text)
      print("----------------------")
      print(prompt)
      print()

      self.request_generator(prompt, index)


  def request_generator(self, prompt, index):
    length = len(prompt)
    device = "cuda:0"
    encoding = self.tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
      outputs = self.model.generate(
          input_ids = encoding.input_ids,
          attention_mask=encoding.attention_mask,
          generation_config = self.generation_config,
      )
    aux = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
    #print(aux)
    print(index)
    
    aux = self.post_processing(aux, length)

    if index == 0:
      respuesta = {
        'id_pregunta':index,
        'var_removed': self.var_removed,
        'prompt': prompt,
        'respuesta':aux
      }
    else:
      if self.ablation:
        respuesta = {
        'id_pregunta':index,
        'var_removed': self.var_removed ,
        'respuesta':aux
        }
      else:
        respuesta = {
          'id_pregunta':index,
          'respuesta':aux
        }
    
    with open(self.jsonl_name, "a") as file:
      json.dump(respuesta, file)
      file.write("\n")

  
  def prompting(self, example, index, fewshot_text):
    if self.eng:
      person_prompt = prompt_generator_english(example, self.exp )
    elif self.ablation:
      person_prompt = example["prompt"]
    elif not self.ablation:
      person_prompt = prompt_generator(example, self.exp, self.wpv, self.red )

  
    if self.fewshot: 
      prompt = fewshot_text + f"""{{
<s>[INST] {self.end}
{person_prompt + self.inst} [/INST]}}"""
    
    elif self.random_fewshot:
      index_list = self.imatrix[index][0:self.nsamples] #list of random 10 indexes that indicates the examples to be used in this case
      fs_examples_prompt = self.random_fewshot_prompt_generator(index_list)
      prompt = fs_examples_prompt + "<s>[INST]" + self.end + person_prompt + self.inst + "[/INST]"
      
    else:
      if self.translate:
        person_prompt = self.translator.translate(person_prompt)

      prompt = f"""{{[INST] <<SYS>>
      {self.task}
      <</SYS>>
      { self.end + person_prompt} [/INST]}}"""

      
    return prompt
  

  def post_processing(self, aux, length):
    #print(aux)
    if self.exp == 3:
      #print(aux.split())
      #cadena_busqueda = "[/INST]}"
      #posicion_inicio = aux.find("[/INST]}")
      #aux = aux.split()[-1]
      #aux = aux[posicion_inicio + len(cadena_busqueda):]
      aux_list = aux.split("[/INST]")
      aux = aux_list[-1]
    elif self.fewshot or self.random_fewshot:
      #print(aux.split())
      aux = aux.split()[-1]
    else:
      aux = aux[length:].strip()
    print(aux)
    return aux


  def fewshot_examples_generator(self,data, random_indexes):
    examples_list = []
    answers_list = []
    if self.exp == 1:
      exp_column = "elec_pres_144_a"
    elif self.exp == 2:
      exp_column = "constitucion_20_a"
    elif self.exp == 3:
      exp_column = "religion_14"

    for index in random_indexes:
      person = data.iloc[index]
      examples_list.append(person["prompt"])
      answers_list.append(person["output"])

    return examples_list, answers_list


  def fewshot_prompt_generator(self, examples_list, answers_list):
    if self.eng:
      s = "First, an example will be given to you, only to familiarize you with the response format, regardless of the chosen option:\n"
    else:
      s = "Primero, se te dará un ejemplo solo para que te familiarices con el FORMATO de respuesta, independiente de la opción elegida:\n"

    if self.eng:
      person_example = "47 year old woman. She lives in the METROPOLITAN region in Chile, in an urban area. Her socioeconomic level is C3 and her educational level is complete technical institute. She does not belong to any indigenous people. She feels close to catholic religion. Regarding politics, on the left-right scale She feels closer to the center, she does not sympathize or feel close to any party, aShe  is not at all interested interested in politics. Now, please answer to the following question taking into account the previous characteristics and bearing in mind that the contextual year is 2022: Which of these ideas best expresses your judgment regarding abortion? Abortion should always be prohibited, abortion should only be allowed in special cases, abortion should be an option for women in any case."
      fewshot_chain = f"""{{<s>[INST] <<SYS>>
    { self.context}
    <</SYS>>
    { s + person_example + self.inst} [/INST] Abortion should always be prohibited.</s>}}"""
    
    else:
      fewshot_chain = f"""{{<s>[INST] <<SYS>>
    { self.context}
    <</SYS>>
    { s + examples_list.pop(0) + self.inst} [/INST] { self.dictionary[int(answers_list.pop(0))] } </s>}}"""
    
      #example_chain = ""
      for example, answer in zip(examples_list,answers_list):
        fewshot_chain += f"""{{
    <s>[INST] { example + self.inst} [/INST] {self.dictionary[int(answer)]} </s>}}"""
    
    return fewshot_chain
  

  #genero un dataset de 100 muestras con los prompts y respuestas que corresponden.
#otro dataset sin esos 100 prompts
#una matriz donde cada fila contiene los 10 ejemplos(indices) para utilizar en cierta persona.
  def random_fewshot_prompt_generator(self, index_list):
    i1=index_list.pop(0)
    example = self.examples_df.iloc[i1]
    person_prompt = prompt_generator(example, self.exp, self.wpv, self.red )

    if self.exp == 3:
      ex = "Primero, se te dará un ejemplo solo para que te familiarices con el FORMATO de respuesta, independiente de la opción elegida:\n"
    else:
      ex = "A continuación se te darán algunos ejemplos:\n"

    fewshot_chain = "<s>[INST] <<SYS>>\n"+ self.context + "\n" + "<</SYS>>\n" + \
      ex +person_prompt +"[/INST]\n"+ \
        self.dictionary[int(self.examples_df.iloc[i1]["religion_14"])] + "</s>"
    
    #example_chain = ""
    for index in index_list:
      example = self.examples_df.iloc[index]
      prompt = prompt_generator(example, self.exp, self.wpv, self.red )
      answer = self.examples_df.iloc[index]["religion_14"]
      fewshot_chain += "<s>[INST]" + prompt + "[/INST]" + self.dictionary[int(answer)] + "</s>"
    return fewshot_chain