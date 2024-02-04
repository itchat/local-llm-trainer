import os
import argparse
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic
from langchain.embeddings import HuggingFaceEmbeddings


os.environ["ANTHROPIC_API_KEY"] = ''

# 创建解析器
parser = argparse.ArgumentParser(description="Chat model with custom persist directory for Chroma.")

parser.add_argument("--dir", type=str, default="db", help="Customize the persist directory for Chroma.")
args = parser.parse_args()

# 初始化开源 embeddings 对象
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 使用自定义的persist_directory参数
docsearch = Chroma(persist_directory=args.dir, embedding_function=embeddings)

model = ChatAnthropic(model="claude-2.1", max_tokens_to_sample=200000, temperature=1)

qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=docsearch.as_retriever(),
                                 return_source_documents=True)
while True:
    lines = []
    while True:
        line = input("INPUT (Enter an empty line to process) -> ")
        if line == "":
            break
        lines.append(line)
    
    if lines:  # Check if the list of lines is not empty
        query = "\n".join(lines)
        result = qa({"query": query})
        print(result['result'])
