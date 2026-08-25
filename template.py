import os 
from pathlib import Path

list_of_files = [

    f"models/",
    f"datas/",
    f"results/",
    f"notebooks/",
    f"src/__init__.py",
    f"src/exception.py",
    f"src/logger.py",
    f"src/rules.py",
    f"src/prompts.py",
    f"src/data_handler.py",
    f"src/agent_1.py",
    f"src/agent_2.py",
    f"src/agent_3.py",
    f"src/config_load.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "config.yaml",
    "main.py" ]

for file_path in list_of_files:
    
    if file_path.endswith("/"):
        os.makedirs(file_path, exist_ok=True)
        print(f"{file_path} is already exists")
        continue
    
    file_path = Path(file_path) 
    filedir, filename = os.path.split(file_path)

    if filedir!= "":
        os.makedirs(filedir, exist_ok=True)
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path, 'w') as file:
            pass
        print(f"{filename} is created in {filedir}")
    else:
        print(f"{filename} is already exists in {filedir} and has some content. Skipping creation.")
        