

C:\Python312\python.exe -V
C:\Python312\python.exe -m ensurepip
C:\Python312\python.exe -m venv .venv

.venv\Scripts\activate

.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
REM .venv\Scripts\pip.exe freeze > current_reqs.txt

REM .venv\Scripts\pip.exe install smolagents pandas langchain langchain-community sentence-transformers datasets python-dotenv rank_bm25 accelerate


REM .venv\Scripts\pip uninstall llama-cpp-python
REM REM .venv\Scripts\pip uninstall huggingface_hub
REM REM .venv\Scripts\pip install pyreadline3
REM .venv\Scripts\pip install transformers
REM .venv\Scripts\pip install ollama
REM .venv\Scripts\pip install pymupdf
REM .venv\Scripts\pip install numpy
REM .venv\Scripts\pip install fastapi
REM .venv\Scripts\pip install uvicorn[standard]
REM .venv\Scripts\pip install torch
REM .venv\Scripts\pip install huggingface_hub[hf_xet]
REM .venv\Scripts\pip install protobuf
REM .venv\Scripts\pip freeze > requirements.txt
REM ollama pull nomic-embed-text
REM ollama pull deepseek-r1:1.5b