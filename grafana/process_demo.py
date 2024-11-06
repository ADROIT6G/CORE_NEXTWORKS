import json
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone, timedelta
import logging
import sys
#NXW testbed
# MQTT settings
# MQTT_BROKER = "10.30.8.80"  
# MQTT_PORT = 31883
# MQTT_USER = "emqx"
# MQTT_PSW = "public"
# MQTT_TOPICA = "monitoring/out"  
# MQTT_TOPICB = "analysis/out"  

# #ttest feed
# MQTT_BROKER = "10.30.8.240"  
# MQTT_PORT = 1883
# MQTT_USERNAME = "emqx"
# MQTT_PASSWORD = "public"
# MQTT_TOPICA = "monitoring/out"  
# MQTT_TOPICB = "analysis/out"  

# #ORO
MQTT_BROKER = "192.168.10.44"  
MQTT_PORT = 31883
MQTT_USER = "emqx"
MQTT_PSW = "public"
MQTT_TOPICA = "monitoring/out"  
MQTT_TOPICB = "analysis/out"  
MQTT_TOPICC = "decision/out" 

# InfluxDB settings
#INFLUXDB_URL = "http://10.30.8.240:8086"  
#INFLUXDB_TOKEN = "dCmCtATPmy1h4njrbFLwqaqBY6k8unOAfAQZ51U8NK7eD4UVjntAwcfV-qdKOef06PWeDEOhdGTngIyXh6swbA=="  
INFLUXDB_TOKEN = "HIw1DkMM6iCVEGJOb6EqOPnVBxxLINVHvoTumZR9hf76Mzg_P7Y0zFDyvKI7ofPniehUs2yHdI_X-rBByTeHfw=="
INFLUXDB_URL = "http://172.28.23.30:8086" 
INFLUXDB_ORG = "eucnc"  
INFLUXDB_BUCKETB = "analysisetsi"  
INFLUXDB_BUCKETA = "dataetsi"  
INFLUXDB_BUCKETC = "decisionetsi" 

# Create InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Convert timestamp string to InfluxDB-friendly format 
def convert_to_ns_and_datetime(timestamp_str):

    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    datetime_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    #print(datetime_str)
    return datetime_str

def handle_topic3_message(msg):
    data = json.loads(msg.payload)
    timestamp=[] 
    value1=[] 
    #print("############################################")
    #print("analysis")
    for entry in data[0]:
        timestampi, value1i = entry
        timestamp.append(timestampi)
        value1.append(float(value1i))

    return timestamp, value1

def handle_topic2_message(msg):
    data = json.loads(msg.payload)
    timestamp=[] 
    value1=[] 
    value2=[]  
    value3=[] 
    value4=[] 
    value5=[]
    #print("############################################")
    #print("analysis")
    for entry in data[0]:
        timestampi, value1i, value2i, value3i, value4i, value5i = entry
        timestamp.append(timestampi)
        value1.append(float(value1i))
        value2.append(float(value2i))
        value3.append(float(value3i))
        value4.append(float(value4i))
        value5.append(float(value5i))

    return timestamp, value1, value2, value3, value4, value5

def on_message(client, userdata, msg):
    if msg.topic == MQTT_TOPICA:
        try:
            #print("############################################")
            #print("monitoring")
            payload = json.loads(msg.payload)
            #print(payload)
            for data in payload:
                datetime_str = convert_to_ns_and_datetime(data["timestamp"])
                point = Point("mqtt_data") \
                    .tag("id", "data") \
                    .field("URLLC_BytesReceived", float(data["URLLC_BytesReceived"])) \
                    .field("URLLC_BytesSent", float(data["URLLC_BytesSent"])) \
                    .field("URLLC_Received_thrp_Mbps", float(data["URLLC_Received_thrp_Mbps"])) \
                    .field("URLLC_Sent_thrp_Mbps", float(data["URLLC_Sent_thrp_Mbps"])) \
                    .field("datetime", datetime_str) 
                write_api.write(INFLUXDB_BUCKETA, INFLUXDB_ORG, point)
                # print("############################################")
                # print("monitoring")
                #print(point)
            #print("Data written to InfluxDB 1")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    elif msg.topic == MQTT_TOPICB:
        try:
            #print("############################################")
            #print("decision")
            timestamp, value1, value2, value3, value4, value5 = handle_topic2_message(msg)

            length = len(timestamp)
            for i in range(length):
                datetime_str = convert_to_ns_and_datetime(timestamp[i])
                point = Point("mqtt_data") \
                    .tag("id", "anfis") \
                    .field("score1", value1[i]) \
                    .field("score2", value2[i]) \
                    .field("score3", value3[i]) \
                    .field("score4", value4[i]) \
                    .field("score5", value5[i]) \
                    .field("datetime", datetime_str) 
                write_api.write(INFLUXDB_BUCKETB, INFLUXDB_ORG, point)
                #print(point)
            #logger.info("Data written to InfluxDB 2")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    elif msg.topic == MQTT_TOPICC:
        try:
            timestamp, value1 = handle_topic3_message(msg)

            length = len(timestamp)       
            for i in range(length):
                datetime_str = convert_to_ns_and_datetime(timestamp[i])
                point = Point("mqtt_data") \
                    .tag("id", "rfc") \
                    .field("value", value1[i]) 
                write_api.write(INFLUXDB_BUCKETC, INFLUXDB_ORG, point)
        except Exception as e:
            logger.error(f"Error processing message: {e}")


def on_connect(client, userdata, flags, rc):
    logger.info(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPICA)
    client.subscribe(MQTT_TOPICB)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info('Translator Started')

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USER, MQTT_PSW)

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_forever()