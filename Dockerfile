FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt update \
    && apt install --no-install-recommends -y libgl1 libglib2.0-0 g++

COPY requirements.txt ./
RUN pip install --no-cache -r requirements.txt

ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt ./
COPY yolo_demo ./yolo_demo

EXPOSE 1883
EXPOSE 8553

CMD ["python", "-m", "yolo_demo"]
