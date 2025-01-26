import os 

def check_model_exist(path:str)-> bool:
    if os.path.exists(path):
        return True
    return False