import argparse
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

# 创建解析器
parser = argparse.ArgumentParser(description="Process documents and customize persist directory for Chroma.")
# 添加--persist_directory参数
parser.add_argument("--dir", type=str, default="db", help="Customize the persist directory for Chroma.")
args = parser.parse_args()

# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader(r'data/', glob='*.*')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=5000)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

# 初始化开源 embeddings 对象
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 将 document 通过开源的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory=args.dir)
docsearch.persist()
