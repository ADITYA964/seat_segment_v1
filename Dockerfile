FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update \
    && apt-get install -y git python3-pip python3.9 

RUN apt-get install ffmpeg libsm6 libxext6 -y
  
WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

RUN mkdir ~/.streamlit

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]



