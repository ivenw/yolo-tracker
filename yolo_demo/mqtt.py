from typing import Optional, Protocol

from paho.mqtt.client import Client


class MqttClient(Protocol):
    def publish(self, topic: str, payload: str) -> None:
        ...


class DummyMqttClient(MqttClient):
    def publish(self, topic: str, payload: str) -> None:
        del topic
        print(payload)


class PahoMqttClient(MqttClient):
    def __init__(self) -> None:
        self.client = Client()

    def publish(self, topic: str, payload: str) -> None:
        self.client.publish(topic, payload)

    def connect(
        self, host: str, port: int, username: Optional[str], password: Optional[str]
    ) -> None:
        if username is not None and password is not None:
            self.client.username_pw_set(username, password)
        elif username is not None:
            self.client.username_pw_set(username)

        self.client.connect(host, port)
