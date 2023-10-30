import torch
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

# Paths and filenames
pdf_file_path = "finance.pdf"
model_path = "FlagAlpha/Llama2-Chinese-13b-Chat-4bit"
embedding_model_name = "FlagAlpha/Llama2-Chinese-13b-Chat-4bit"

# Constants for instruction and system prompts
BEGIN_INST, END_INST = "[INST]", "[/INST]"
BEGIN_SYS, END_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant... (your system prompt here)"""

# Function to generate a combined prompt
def create_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    full_system_prompt = BEGIN_SYS + system_prompt + END_SYS
    full_prompt = BEGIN_INST + full_system_prompt + instruction + END_INST
    return full_prompt

# Function to generate output by applying the model to a prompt
def generate_output(model, tokenizer, prompt):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        decoded_outputs = decoded_outputs.replace('</s>', '').strip()
    return decoded_outputs

# Function to parse and format text output
def parse_and_format(text):
    wrapped_text = textwrap.fill(text, width=100)
    print(wrapped_text + '\n\n')

# Main function
def main():
    # Load PDF and split into chunks
    pdf_loader = PyPDFLoader(pdf_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    splitted_docs = text_splitter.split_documents(pdf_loader.load())
    
    # Create embeddings and vector store
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cuda"},
    )
    vector_store = Chroma(
        collection_name='llama2_demo',
        embedding_function=embeddings,
        persist_directory='./'
    )
    vector_store.add_documents(splitted_docs)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    # Create a language model pipeline
    generation_pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        max_new_tokens=512,
        do_sample=True,
        top_k=30,
        num_return_sequences=1,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=generation_pipe, model_kwargs={'temperature': 0})
    
    # Create prompts and chains
    system_prompt = "You are an advanced assistant that excels at translation. "
    instruction = "Convert the following text from English to French:\n\n{text}"
    full_prompt = create_prompt(instruction, system_prompt)
    print(full_prompt)
    
    prompt_template = PromptTemplate(template=full_prompt, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    # Run the model and output results
    input_text = "how are you today?"
    output_text = llm_chain.run(input_text)
    parse_and_format(output_text)

if __name__ == '__main__':
    main()
