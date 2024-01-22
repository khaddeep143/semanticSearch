FROM ubuntu

RUN apt update
RUN apt install python3-pip -y
RUN pip3 install FLASK
RUN pip3 install llama_index
RUN pip3 install waitress
RUN pip3 install safetensors
RUN pip3 install setuptools
RUN pip3 install transformers
RUN pip3 install transformers
RUN pip3 install anaconda
RUN pip3 install torch
RUN pip3 install openai
RUN pip install pypdf -q -U
RUN pip install pypdf
RUN pip install sentence-splitter
RUN pip install sentence_transformers
RUN pip install pymilvus
#RUN pip3 install poetry


WORKDIR /app

COPY . .

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]

#docker run -d -p 5000:5000 flask
#docker build -t flask:0.0.1

#poetry export | tee requirements.txt
