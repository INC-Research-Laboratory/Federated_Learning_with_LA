import socket
import numpy as np
import os
import sys
import argparse
from Edge_class import Edge
import time
import threading

# Parameter
parser = argparse.ArgumentParser()
parser.add_argument("--total_device_num", type=int, default="2", help="edge_num and learning_agent_num total num")  # 성능측정에 참여하는 총 device 개수 device = la + edge
parser.add_argument("--learning_agent_num", type=int, default="1", help="learning agent num")   # 성능 측정에 참여하는 Learning agnet의 개수
parser.add_argument("--time_slice", type=int, default="20", help="learning agent num")   # 몇 분에 한 번 성능측정을 할 지


parser.add_argument("--target_device_os", type=str, default="window", help="target_device_os")  

opt = parser.parse_args()

LaInfo_Dir = os.getcwd()


if __name__ == "__main__":
    Edge = Edge(opt.learning_agent_num, opt.total_device_num, LaInfo_Dir, opt.target_device_os)
    
    if opt.learning_agent_num != 0:
        Edge.La_bind_socket1()
    else:
        Edge.Server_connect_socket1()

    

    
    

