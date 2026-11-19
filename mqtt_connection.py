"""MQTT connection settings shared by the bedside and station launchers."""
import os
import re

DEPARTMENTS = [
    ("ICU", "Intensive Care Unit"), ("CCU", "Coronary Care Unit"),
    ("NCU", "Neurological Care Unit"), ("NICU", "Neonatal Intensive Care Unit"),
    ("PICU", "Pediatric Intensive Care Unit"), ("SICU", "Surgical Intensive Care Unit"),
    ("Trauma / ER", "Emergency & Trauma"), ("General Ward", "General & Step-Down Ward"),
]


def valid_device_id(value):
    return (isinstance(value, str) and value != "nurse_stations"
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) is not None)


def mqtt_setting(name):
    # Existing installations can migrate their environment one setting at a time.
    return os.environ.get(f"MQTT_{name}", os.environ.get(f"HIVEMQ_{name}"))


def broker_host():
    return mqtt_setting("HOST") or "127.0.0.1"


def local_host(host):
    return host in {"127.0.0.1", "localhost", "::1"}


def broker_port(host):
    return int(mqtt_setting("PORT") or (1884 if local_host(host) else 8883))


def mqtt_client(client_id, host, port):
    import paho.mqtt.client as mqtt

    args = {"client_id": client_id}
    if hasattr(mqtt, "CallbackAPIVersion"):
        args["callback_api_version"] = mqtt.CallbackAPIVersion.VERSION2
    client = mqtt.Client(**args)
    user = mqtt_setting("USERNAME")
    password = mqtt_setting("PASSWORD")
    if bool(user) != bool(password):
        raise ValueError("Set both MQTT_USERNAME and MQTT_PASSWORD.")
    if not local_host(host) and not user:
        raise ValueError("A remote MQTT broker requires a username and password for this device or station.")
    if not local_host(host) or port == 8883 or user or mqtt_setting("CA_FILE"):
        client.tls_set(ca_certs=mqtt_setting("CA_FILE") or None)
    if user:
        client.username_pw_set(user, password)
    return client
