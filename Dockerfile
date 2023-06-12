FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt update \
    && apt install --no-install-recommends -y libgl1 libglib2.0-0 g++
    # && apt install --no-install-recommends -y libgl1-mesa-glx  libpython3-dev gnupg g++
# RUN apt upgrade --no-install-recommends -y openssl tar

# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# RUN pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
COPY requirements.txt ./
RUN pip install --no-cache -r requirements.txt

ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt ./
COPY yolo_demo ./yolo_demo

EXPOSE 1883
EXPOSE 8553

ENV RTSP_STREAM="rtsp://192.168.10.109:8554/live.sdp"
ENV MQTT_HOST="host.docker.internal"
ENV MQTT_PORT="1883"
ENV MQTT_TOPIC="yolo"
ENV TRACKING_AREAS='[{"tag": "test", "area": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]}]'

CMD ["python", "-m", "yolo_demo"]
