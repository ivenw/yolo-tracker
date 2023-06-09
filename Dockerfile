FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++
RUN apt upgrade --no-install-recommends -y openssl tar

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
COPY requirements.txt ./
RUN pip install --no-cache -r requirements.txt

ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt ./
COPY yolo_demo ./yolo_demo

EXPOSE 1883
EXPOSE 8553

CMD ["python", "yolo_demo"]
