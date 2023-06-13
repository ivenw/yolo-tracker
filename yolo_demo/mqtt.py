from typing import Optional, Protocol, Union

from paho.mqtt.client import Client


Payload = Union[str, bytes]


class MqttClient(Protocol):
    def publish(self, topic: str, payload: Payload) -> None:
        ...


class DummyMqttClient(MqttClient):
    def publish(self, topic: str, payload: Payload) -> None:
        del topic
        print(payload)


class PahoMqttClient(MqttClient):
    def __init__(self) -> None:
        self.client = Client()

    def publish(self, topic: str, payload: Payload) -> None:
        self.client.publish(topic, payload)

    def connect(
        self, host: str, port: int, username: Optional[str], password: Optional[str]
    ) -> None:
        if username is not None and password is not None:
            self.client.username_pw_set(username, password)
        elif username is not None:
            self.client.username_pw_set(username)

        self.client.connect(host, port)
