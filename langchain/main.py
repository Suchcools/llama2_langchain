# 兼容不同行业 不同问题的模板

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
import chromadb
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import sys
from template import get_template

# 导入模型与配置

sys.path.append(os.path.dirname(os.path.realpath('__file__'))+"/../")
import config.conf as cfg
from chromadb.config import Settings

# Define the folder for storing database
PERSIST_DIRECTORY = cfg.PERSIST_DIRECTORY
# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

embeddings_model_name = cfg.EMBEDDINGS_MODEL_NAME
persist_directory = cfg.PERSIST_DIRECTORY
model_path = cfg.MODEL_PATH
model_n_ctx = cfg.MODEL_N_CTX
model_n_batch = cfg.MODEL_N_BATCH
target_source_chunks = 4
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)

## 加载模型

from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
model = HuggingFacePipeline.from_model_id(model_id=model_path,
        task="text-generation",
        device=0,
        pipeline_kwargs={
            "max_new_tokens": 4096, 
            "do_sample": True,
            "temperature": 0.2,
            "top_k": 1,
            "top_p": 0.9,
            "repetition_penalty": 1.1},
        model_kwargs={
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True}
        )

### 文档查询 去向量数据库匹配相关代码
doc_query, refine_query, summary_query, response_schemas= get_template("人工智能","产业规模") # 行业 和 三级问题


### 问题检索
retriever = db.as_retriever(search_kwargs={"k": 20})
docs = retriever.get_relevant_documents(doc_query)


### 回答策略 refine map-reduce

qa_chain_refine = RetrievalQA.from_chain_type(
    model,
    retriever=db.as_retriever(search_kwargs={"k": 10}),
    chain_type="refine"
)
 
result = qa_chain_refine({"query": refine_query})
result["result"]

### 数据解析格式化返回

que =result["result"]

initial_qa_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。\n"
    "<</SYS>>\n\n"
    "以下为背景知识：\n"
    "{context_str}"
    "\n"
    "{format_instructions}"
    "\n"
    "请根据以上背景知识, 回答这个问题：{question}。"
    " [/INST]"
)


output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template=initial_qa_template,
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions,"context_str":que}
)
_input = prompt.format_prompt(question=summary_query)
output = model(_input.to_string())
print(output)
res = output_parser.parse(output)
res
 