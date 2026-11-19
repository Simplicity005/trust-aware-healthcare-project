"""MQTT connection settings shared by the bedside and station launchers."""
import os
import re

import paho.mqtt.client as mqtt

DEPARTMENTS = [
    ("ICU", "Intensive Care Unit"), ("CCU", "Coronary Care Unit"),
    ("NCU", "Neurological Care Unit"), ("NICU", "Neonatal Intensive Care Unit"),
    ("PICU", "Pediatric Intensive Care Unit"), ("SICU", "Surgical Intensive Care Unit"),
    ("Trauma / ER", "Emergency & Trauma"), ("General Ward", "General & Step-Down Ward"),
]


def valid_device_id(value):
    return (isinstance(value, str) and value != "nurse_stations"
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) is not None)


def broker_host():
    return os.environ.get("HIVEMQ_HOST") or "127.0.0.1"


def local_host(host):
    return host in {"127.0.0.1", "localhost", "::1"}


def broker_port(host):
    return int(os.environ.get("HIVEMQ_PORT") or (1884 if local_host(host) else 8883))


def mqtt_client(client_id, host, port):
    args = {"client_id": client_id}
    if hasattr(mqtt, "CallbackAPIVersion"):
        args["callback_api_version"] = mqtt.CallbackAPIVersion.VERSION2
    client = mqtt.Client(**args)
    user = os.environ.get("HIVEMQ_USERNAME")
    password = os.environ.get("HIVEMQ_PASSWORD")
    if bool(user) != bool(password):
        raise ValueError("Set both HIVEMQ_USERNAME and HIVEMQ_PASSWORD.")
    if not local_host(host) and not user:
        raise ValueError("Remote HiveMQ requires a username and password for this device or station.")
    if not local_host(host) or port == 8883 or user:
        client.tls_set(ca_certs=os.environ.get("HIVEMQ_CA_FILE") or None)
    if user:
        client.username_pw_set(user, password)
    return client
