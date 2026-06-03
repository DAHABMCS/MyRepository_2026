
from datetime import datetime

def unix_to_readable(unix_timestamp):
    """Convert Unix timestamp to a datetime object."""
    date_time = datetime.fromtimestamp(unix_timestamp / 1000.0)
    readable_date_time = date_time.strftime('%Y-%m-%d %H:%M:%S')
    return readable_date_time

def convert_to_unix_timestamp(date_str, date_format="%Y-%m-%dT%H:%M:%S.%fZ"):
    dt = datetime.strptime(date_str, date_format)
    return int(dt.timestamp() * 1000)


def load_config(file_path='config1.json'):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Function to enter a trade (buy)
def enter_trade(exchange, symbol, amount) :
    try :
        order = exchange.create_market_buy_order(symbol, amount)
        print(f"Buy Order Placed: {order}")
        return order
    except Exception as e:
        print(f"Error Placing Buy Order: {e}")
        return None

# Function to exit a trade (sell)
def exit_trade(exchange, symbol, amount):
    try:
        order = exchange.create_market_sell_order(symbol, amount)
        print(f"Sell Order Placed: {order}")
        return order
    except Exception as e:
        print(f"Error Placing Sell Order: {e}")
        return None


def load_config(file_path='config.json'):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def generate_signature(timestamp, method, request_path, body, secret):
    message = timestamp + method + request_path + body
    mac = hmac.new(bytes(secret, 'utf-8'), bytes(message, 'utf-8'), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode('utf-8')


def convert_to_unix_timestamp(date_str, date_format="%Y-%m-%dT%H:%M:%S.%fZ"):
    dt = datetime.strptime(date_str, date_format)
    return int(dt.timestamp() * 1000)

import base64
import hashlib
import hmac
import json
import time

def load_config():
    with open('../config.json') as f:
        config = json.load(f)
    return config


def generate_signature(timestamp, method, request_path, body, secret_key):
    message = timestamp + method + request_path + body
    print(request_path)
    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod=hashlib.sha256)
    d = mac.digest()
    return base64.b64encode(d)


def convert_to_unix_timestamp(date_str):
    return int(time.mktime(time.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")) * 1000)
