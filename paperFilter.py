from mlx_lm import load, generate
from ast import literal_eval
import re

class paperFilter(object):
    
    def __init__(self, model = "mlx-community/Meta-Llama-3-8B-Instruct-4bit", adaptor = None, mod_con = {}) -> None:
        self.model, self.tokenizer = load(model, adapter_path = adaptor, model_config = mod_con)

    def filter_papers(self, dataframe):
        titles = dataframe['title']
        title_filter = self.__check_titles(titles)
        filtered_df = dataframe[title_filter]
        return filtered_df
    
    def __check_titles(self, titles):
        prompt = "For this title, write True if it is new, innovative, and interesting and False otherwise. Only output the value True or False."
        role = "You are screening research papers, be very selective. You're focus is cutting-edge research. Only output should be True or False."
        temp = 0.1
        filters = []
        for title in titles:
            try:
                input = str(titles) + "\n\n" + prompt
                prompt = self.tokenizer.apply_chat_template([{'role': 'system', 'content': role}, {'role': 'user', 'content': input}], tokenize=False, add_generation_prompt=True)
                response = generate(self.model, self.tokenizer, prompt=prompt, temp=temp, verbose=False)
                if "True" in response and "False" not in response:
                    filters.append(True)
                elif "False" in response and "True" not in response:
                    filters.append(False)
                else:
                    raise Exception
            except:
                pass
        return filters

        
