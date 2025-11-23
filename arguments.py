import json


class Arguments():
    def __init__(self, args_path):
        args_dict = self.read_file(args_path)
        for k, v in args_dict.items():
            setattr(self, k, v)
    
    def read_file(self, args_path):
        with open(args_path, 'r', encoding='utf-8') as f:
            return json.load(f)

