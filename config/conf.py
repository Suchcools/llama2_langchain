# 定义配置变量
import os

#向量数据库
PROJECT_PATH = os.path.dirname(os.path.realpath('__file__'))

TARGET_DATABASE = f"人工智能"
PERSIST_DIRECTORY = f"{PROJECT_PATH}/DB/db"
SOURCE_DIRECTORY = f"{PROJECT_PATH}/DB/Process/source"


#模型
MODEL_PATH = f"{PROJECT_PATH}/checkpoints/chinese-alpaca-2-13b-16k"
MODEL_N_CTX = 4096
MODEL_N_BATCH = 512
EMBEDDINGS_MODEL_NAME= f"{PROJECT_PATH}/checkpoints/paraphrase-multilingual-MiniLM-L12-v2"