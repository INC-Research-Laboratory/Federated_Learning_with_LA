import socket
import numpy as np
import os
import sys
import argparse
from Server_class import Server
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--edge_device_num", type=int, default="2", help="edge num")  
opt = parser.parse_args()

LaInfo_Dir = os.getcwd()


if __name__ == "__main__":
    Server = Server(opt.edge_device_num, opt.round)
    Server.Edge_bind_socket1()
