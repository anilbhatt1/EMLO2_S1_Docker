FROM python:3.9.17-slim

WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]

CMD['mobilenetv3_large_100', 'https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/test/12/image/image.jpg']