import socket
import sys
import numpy as np
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms
import pickle
import struct
from collections import OrderedDict
import copy
import time
import threading
import select
import re

SERVER_IP = "Server IP"
SERVER_PORT1 = 50001  # 항상 열려있는 포트(Server -> Edge)
SERVER_PORT2 = 60001    # 주기적으로 열리는 포트(Server -> Edge)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = self.drop2D(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x)

model = ConvNet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConvNet().to(device)
torch.manual_seed(777)



class Server(ConvNet):
    def __init__(self, Edge_num, round):
        super().__init__()
        self.Edge_num = Edge_num
        self.Edge_socket1 = []
        self.Edge_socket2 = []
    
        self.port2_Condition = []
        self.port2_flag = None
        self.port2_flag2 = True
        
        self.control_message = None
        self.model = []
        self.update_model = []
        self.accuracy= []

        self.model_semaphore = threading.Semaphore(self.Edge_num)
        self.accuracy_semaphore = threading.Semaphore(self.Edge_num)



    def handle1_Edge(self, EDGE_SOCKET, EDGE_ID):
        print(f"Edge {EDGE_ID} connection : {EDGE_SOCKET.getpeername()}")
        self.Edge_socket1.append(EDGE_SOCKET)

        while True:
            EDGE_SOCKET.sendall(b"Server [always open] port")
            message = EDGE_SOCKET.recv(1024).decode()

            print(f"[{EDGE_ID}] {EDGE_SOCKET.getpeername()} : {message}")


            if "train measure" in message:
                if message == 'connection successtrain measure' or message == "train measureconnection success":
                    pattern = r'train measure'
                    result = re.search(pattern, message)
                    extracted_text = result.group()
                    self.port2_Condition.append(extracted_text)

                else:
                    self.port2_Condition.append(message)
                self.port2_flag = None
                self.port2_flag2 = None




            if ("Send model" in message) or ("Receive model" in message):
                if message == 'connection successReceive model':
                    pattern = r'Receive model'

                    result = re.search(pattern, message)
                    extracted_text = result.group()
                    self.port2_Condition.append(extracted_text)
                elif message == 'connection successSend model':
                    pattern = r'Send model'

                    result = re.search(pattern, message)
                    extracted_text = result.group()
                    self.port2_Condition.append(extracted_text)
                else:
                    self.port2_Condition.append(message)

                print(self.port2_Condition)
            

            if len(self.port2_Condition) == self.Edge_num:
                
                if self.port2_Condition[self.Edge_num-1] == "train measure":
        
                    if self.port2_flag == None:
                        print("Edge measure")
                        self.control_message = "Send model"
                        self.port2_Condition.clear()
                    
                    for i in range(self.Edge_num):
                        self.Edge_socket1[i].send("server port2 open".encode())
                    Edge_bind_socket2_thread = threading.Thread(target=self.Edge_bind_socket2)
                    Edge_bind_socket2_thread.start()


                
                elif self.port2_Condition[self.Edge_num-1] == "Send model":
                
                    if self.port2_flag == True or self.port2_flag2 == True:
                        print("\n[FIRST SEND MODEL]\n")
                        self.control_message = "Send model"
                        self.port2_Condition.clear()

                        for i in range(self.Edge_num):
                            self.Edge_socket1[i].send("server port2 open".encode())

                        self.port2_flag = False
                        self.port2_flag2 = False
                        Edge_bind_socket2_thread = threading.Thread(target=self.Edge_bind_socket2)
                        Edge_bind_socket2_thread.start()

                elif self.port2_Condition[self.Edge_num-1] == "Receive model":
                
                    if self.port2_flag == False or self.port2_flag2 == False:
                        print("\n[RECEIVE MODEL]\n")
                        self.control_message = "Receive model"
                        self.port2_Condition.clear()
                        for i in range(self.Edge_num):
                            self.Edge_socket1[i].send("server port2 open".encode())

                        self.port2_flag = True
                        self.port2_flag = True
                        Edge_bind_socket2_thread = threading.Thread(target=self.Edge_bind_socket2)
                        Edge_bind_socket2_thread.start()
            
            if (self.update_model != [] and self.accuracy_average() <= 91.0):
                if self.port2_flag == True:
                    print(f"Accuracy average = {self.accuracy_average()}")
                    self.accuracy.clear()
                    self.control_message = "resending"            
                        
                    for i in range(self.Edge_num):
                        self.Edge_socket1[i].send("server socket2 reopen".encode())
                    print("\n[RESENDING MODEL]\n")
                    
                    self.port2_flag = False
                    Edge_bind_socket2_thread = threading.Thread(target=self.Edge_bind_socket2)
                    Edge_bind_socket2_thread.start()

            time.sleep(3)

    def Edge_bind_socket1(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((SERVER_IP, SERVER_PORT1))
        server_socket.listen(self.Edge_num)

        print("Server Start...")
        EDGE_ID = 0

        while True:
            EDGE_SOCKET, EDGE_ADDRESS = server_socket.accept()
            EDGE_ID += 1

            EDGE_thread1 = threading.Thread(target=self.handle1_Edge, args=(EDGE_SOCKET, str(EDGE_ID)))
            EDGE_thread1.start()
    

    def Edge_bind_socket2(self):
        if self.port2_flag == None:
            self.port2_flag = True
        server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket2.bind((SERVER_IP, SERVER_PORT2))
        server_socket2.listen(self.Edge_num)
        EDGE_ID2 = 0
        while True:
            print("server port2 open")
            EDGE_SOCKET2, _ = server_socket2.accept()
            self.Edge_socket2.append(EDGE_SOCKET2)
            EDGE_ID2 += 1
            
            print(f"self.Edge_socket2 = {self.Edge_socket2}")

            if len(self.Edge_socket2) == self.Edge_num:
                break


        if self.control_message == "Send model" or self.control_message == "resending":
            self.model_sending()
            

        elif self.control_message == "Receive model":
            self.model_receiving()
    
    
    def model_sending(self):

        for i in range(self.Edge_num):
            self.Edge_socket2[i].send("send".encode())
            #checking
        

            if self.control_message == "Send model":
                weight = model.state_dict().items()
            elif self.control_message == "resending":   
                weight = self.update_model[0]
                #self.update_model.clear()

            dict_weight = dict(weight)
            send_weight = pickle.dumps(dict_weight)
            self.Edge_socket2[i].send(len(send_weight).to_bytes(4, byteorder='big'))                    
            self.Edge_socket2[i].send(send_weight)
        self.update_model.clear()
        self.Edge_socket2.clear()
        #========================= checking ==========================#
        tensor_value = dict_weight['conv1.weight']                       
        print(f"sending model = {tensor_value[0][0][0][0:3]}\n")                          
        #=============================================================#

    def model_receiving(self):
        
        for i in range(self.Edge_num):
            self.Edge_socket2[i].send("rece".encode())

            self.model_semaphore.acquire()
            self.accuracy_semaphore.acquire()
            time.sleep(3)
            data_size = struct.unpack('>I', self.Edge_socket2[i].recv(4))[0]
            received_payload = b""
            remaining_payload_size = data_size
            while remaining_payload_size != 0:
                received_payload += self.Edge_socket2[i].recv(remaining_payload_size)
                remaining_payload_size = data_size - len(received_payload)
            self.model.append(pickle.loads(received_payload))
                
            self.Edge_socket2[i].sendall(b"accuracy")
            self.accuracy.append(float(self.Edge_socket2[i].recv(1024).decode()))
            self.model_semaphore.release()
            self.accuracy_semaphore.release()
        
        
        self.aggregation_model_update()
        self.Edge_socket2.clear()


    def aggregation_model_update(self):
        self.model_semaphore.acquire()
        self.accuracy_semaphore.acquire()

        if self.Edge_num == 1:
            w_avg1 = copy.deepcopy(self.model[0])
            self.model.clear()

            self.update_model.append(w_avg1)

        elif self.Edge_num == 2:
            w_avg1 = copy.deepcopy(self.model[0])
            w_avg2 = copy.deepcopy(self.model[1])
            self.model.clear()
    
            self.update_model.append(w_avg1)
        print(f"After update = {self.update_model[0]['conv1.weight'][0][0][0][0:3]}")

        self.model_semaphore.release()
        self.accuracy_semaphore.release()

    def accuracy_average(self):
        if len(self.accuracy) == 0: 
            return 0
        else:
            total = sum(self.accuracy)
            acc_average = total/len(self.accuracy)
    
            return acc_average
