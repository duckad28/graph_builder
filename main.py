from ie_llm import *
import pandas as pd
import json
import re

task = "ie"
data_path = "./data/full_data.csv"
output_path = "./output/output.jsonl"


def preprocessing_text(text):
        text = re.sub("\xad|\u200b|\ufeff", "", text)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # geometric shapes extended
            "\U0001F800-\U0001F8FF"  # supplemental arrows-C
            "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
            "\U0001FA00-\U0001FA6F"  # chess, symbols, etc.
            "\U0001FA70-\U0001FAFF"  # symbols, objects, emoji
            "\U00002700-\U000027BF"  # dingbats
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub("", text)
        text = re.sub(r"[“”\"\'\*\~\|<>•◆◇★☆…→←↑↓※✔️✅❌#@]+", "", text)
        pattern = r'(?:(?<=\d)|(?<=\dm))[xX](?=\d)'
        text = re.sub(pattern, ' x ', text)
        return text

def extract_infor(input_file, output_file, task = 'ie'):
    data = pd.read_csv(input_file, encoding='utf-8')
    ie_model = IE_LLM(task=task)
    with open(output_file, 'w', encoding='utf-8') as f:
        for content in data['content']:
            try:
                content = preprocessing_text(content)
                extracted_output = ie_model.do_task(content=content)
                output = extracted_output
                if task != 'ie':
                    output = extracted_output[1]
                json_line = json.dumps(output)
                f.write(json_line + "\n")
            except Exception as e:
                print(e)

extract_infor(input_file=data_path, output_file=output_path, task=task)

        