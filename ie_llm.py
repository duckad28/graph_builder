
from unsloth import FastLanguageModel
import ast
import re

class IE_LLM():
    label_entity_list = [
        'property_type',
        'house_design',
        'area_m2',
        'internal_amenities',
        'legal_status',
        'direction',
        'total_price',
        'full_address',
        'near_facilities',
        'contact_name',
        'contact_phone'
    ]
    label_relation_list = [
        'has_design',
        'has_area',
        'has_internal_amenity',
        'has_legal',
        'has_direction',
        'has_price',
        'located_in',
        'near_to',
        'has_contact_person',
        'has_contact_phone'
    ]

    def __init__(self, task='ie', ner_model = "duckad/llama-ner", re_model = "duckad/llama-re-v2"):
        self.tokenizer_re = None
        self.model_re = None
        self.tokenizer_ner = None
        self.model_ner = None
        self.task = task.strip().lower()
        self.ner_model_name = ner_model
        self.re_model_name = re_model
        self.load_model()
   
    def do_task(self, content, extracted_ner = None):
        if self.task == 'ner':
            return self.extract_ner(content=content)
        if self.task == 're':
            return self.extract_re(content=content, extracted_ner=extracted_ner)
        if self.task == 'ie':
            return self.extract_ie(content=content)
        
    # load extract model
    def load_model(self):
        config = {
            'max_seq_length': 4096,
            'dtype': None,
            'load_in_4bit': True
        }
        try:
            if self.task == 'ner':
                self.load_model_ner(config=config, model_ner_name = self.ner_model_name)
            if self.task == 're':
                self.load_model_re(config=config, model_re_name = self.re_model_name)
            if self.task == 'ie':
                self.load_model_ner(config=config, model_ner_name = self.ner_model_name)
                self.load_model_re(config=config, model_re_name = self.re_model_name)
        
        except Exception as e:
            print(e)


    def load_model_ner(self, model_ner_name, config = None):
        self.model_ner, self.tokenizer_ner = FastLanguageModel.from_pretrained(
            model_name = model_ner_name,
            **config
        )
        FastLanguageModel.for_inference(self.model_ner) # Enable native 2x faster inference
    

    def load_model_re(self, model_re_name, config = None):
        self.model_re, self.tokenizer_re = FastLanguageModel.from_pretrained(
            model_name = model_re_name,
            **config
        )
        FastLanguageModel.for_inference(self.model_re) # Enable native 2x faster inference 
    

    # extract information
    def extract_ie(self, content):
        output = {}
        output['ner'] = {}
        output['re'] = {}
        output['jsonl'] = {}
        output['jsonl']['input'] = {}
        output['jsonl']['input']['content'] = content 
        # Extract
        extract_ner_output = self.extract_ner(content=content)
        extract_re_output = self.extract_re(content=content, extracted_ner=extract_ner_output[1])
        output['jsonl']['input']['extracted_ner'] = extract_ner_output[1]
        output['jsonl']['output'] = extract_re_output[1]
        output['ner'] = extract_ner_output[0]
        output['re'] = extract_re_output[0]
        return output    


    def extract_ner(self, content):
        output = []
        try:
            entities = {label_name: [] for label_name in self.label_entity_list}
            match = None
            count = 0
            while not match and count < 10:
                output_extracted_ner = self.llm_extract_ner(content = content, model=self.model_ner, tokenizer=self.tokenizer_ner)
                match = re.search(r'(\{[\s\S]*["\']entities["\']:[\s\S]*\})', output_extracted_ner)
                count += 1
            if match:
                output_extracted_ner = match.group(1)
            else:
                output_extracted_ner = "{'entities': []}"
            extracted_ner = ast.literal_eval(output_extracted_ner)
            for entity in extracted_ner['entities']:
                if entity['type'] in self.label_entity_list:
                    entities[entity['type']].append(entity['text'])
            output.append(entities)
            output.append(extracted_ner)
        except Exception as e:
            print("NER exception: ", e)
        if len(output) != 2:
            output.append({label_name: [] for label_name in self.label_entity_list})
            output.append(ast.literal_eval("{'entities': []}"))
        return output


    def extract_re(self, content, extracted_ner = None):
        output = []
        try:
            real_estates = []
            match = None
            count = 0
            while(not match and count < 10):
                output_extracted_re = self.llm_extract_re(content= content, extracted_entities=extracted_ner, model= self.model_re, tokenizer=self.tokenizer_re)
                match = re.search(r'(\{[\s\S]*["\']real_estates["\']:[\s\S]*\})', output_extracted_re)
                count += 1
            if match:
                output_extracted_re = match.group(1)   
            else:
                output_extracted_re = "{'real_estates': []}"
            extracted_re = ast.literal_eval(output_extracted_re)
            for real_estate in extracted_re['real_estates']:
                relations = {label_name: [] for label_name in self.label_relation_list}
                for relation in real_estate['relations']:
                    if relation['relation'] in self.label_relation_list:
                        relations[relation['relation']].append(relation['tail'])
                real_estates.append(relations)
            output.append(real_estates)
            output.append(extracted_re)
        except Exception as e:
            print("RE exception: ", e)
        if len(output) != 2:
            output.append([])
            output.append(ast.literal_eval("{'real_estates': []}"))
        return output


    def llm_extract_ner(self, content, model, tokenizer):
        instruction = "Extract named entities from the given Vietnamese real estate text.\nAllowed entity type: [property_type, house_design, area_m2, internal_amenities, direction, legal_status, total_price, full_address, near_facilities, contact_name, contact_phone].\nReturn in JSON format."
        text = f"{instruction}\n\nInput: {content}\n\nOutput:"
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        _ = model.generate(**inputs, max_new_tokens = 4096, pad_token_id = tokenizer.eos_token_id)
        result = tokenizer.decode(_[0], skip_special_tokens=True)
        return result.split("Output: ")[1]


    def llm_extract_re(self, content, extracted_entities, model, tokenizer):
        instruction = "Extract relations between extracted entities of each real estate from the given Vietnamese real estate text. Return in JSON format."
        input = {}
        input['content'] = content
        if extracted_entities:
            input['extracted_ner'] = extracted_entities
        text = f"{instruction}\n\nInput: {input}\n\nOutput:"
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        _ = model.generate(**inputs, max_new_tokens = 4096, pad_token_id = tokenizer.eos_token_id)
        result = tokenizer.decode(_[0], skip_special_tokens=True)
        return result.split("Output: ")[1]

