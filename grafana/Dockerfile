FROM python:3.8.3-slim-buster

RUN apt-get update -y && apt-get install -y python3-pip python-dev
COPY ./process_demo.py /process_demo.py
COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
CMD [ "python3", "/process_demo.py"]
