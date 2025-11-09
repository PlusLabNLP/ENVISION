import csv
import os
import sys
import re
import string
import json
import xml.etree.ElementTree as ET
from IPython.display import Image
import random
import numpy as np
import glob
from termcolor import colored
import pickle
import copy
import pprint
from collections import OrderedDict


def red_text(text):
    return "\x1b[31m" + text + "\x1b[0m"

def blue_text(text):
    return "\x1b[94m" + text + "\x1b[0m"

def green_text(text):
    return "\x1b[32m" + text + "\x1b[0m"

def magenta_text(text):
    return "\x1b[35m" + text + "\x1b[0m"

def yellow_text(text):
    return "\x1b[93m" + text + "\x1b[0m"

def bg_yellow_text(text):
    return "\x1b[43m" + text + "\x1b[0m"

def bg_magenta_text(text):
    return "\x1b[45m" + text + "\x1b[0m"

def bg_green_text(text):
    return "\x1b[46m" + text + "\x1b[0m"

def bg_blue_text(text):
    return "\x1b[44m" + text + "\x1b[0m"

def bold_text(text):
    return "\x1b[1;128m" + text + "\x1b[0m"

def title_text(text):
    return "\x1b[97m" + text + "\x1b[0m"


def read_jsonl_file(file_path):
    inf = open(file_path)
    data = []
    for line in inf:
        datum = json.loads(line.strip())
        data.append(datum)
    return data
