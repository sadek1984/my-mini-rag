# mini-rag
This is a minimal implementation of the RAG model for Excel files retrieval.
## Requirements ›
- Python 3.8 or later
#### Install Python using MiniConda
1) Download and install MiniConda from [here] (https://docs. anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n mini-rag python=3.8
```
3) Activate the environment:
```bash
$ conda activate mini-rag-app
```
## Installation
### Install the required packages
```bash
$ pip install -r requirements.txt , pip install -r src/requirements.txt
```
### Setup the environment variables
```bash
$ cp .env.example .env
```
Set your environment variables in the `.env` file. Like `OPENAI_API_KEY` value.
## Run Docker Compose Services

```bash
$ cd docker
$ cp .env.example. env
update '.env with your

## Run the FastAPI server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001

uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
``` 
## POSTMAN Collection
origin  git@github.com:yourusername/repo.git(fetch)


Download the POSTMAN collection from [/assets/mini-rag-app.postman_collection.json](/ assets/mini-rag-app.postman_collection.json)
# fastapi boilerplate github --> use template from others

*** for devlopmenet only not used in production {
# Show all containers
docker ps -a
# If there are containers, remove them
docker rm $(docker ps -aq)
ولو عايز تحذف حتى الـ running containers كمان، لازم توقفهم الأول:
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)}

# push to github
git push -u origin <your-branch-name>

# ollama serve from cli
/usr/local/bin/ollama serve
# run serve in colab background
!nohup ollama serve & 
!sleep 5 && tail /content/nohup.out
# to run colab server on local machine
ngrok


