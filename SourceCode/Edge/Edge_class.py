import socket
import time
import pickle
import struct
import numpy as np
import os
import random
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transfroms
import torchvision
import copy
import time
import subprocess
import re

EDGE_IP = "Edge ip"
SERVER_IP = "Server ip"
SERVER_PORT1 = 50001   # 항상 열려있는 포트(Edge -> Server)
SERVER_PORT2 = 60001    # 주기적으로 열리는 포트 1(Edge -> Server)
EDGE_PORT1 = 20001    # 항상 열려있는 포트(Edge -> La)
EDGE_PORT2 = 30001      # 주기적으로 열리는 포트 1(Edge -> La)
EDGE_PORT3 = 35001      # 주기적으로 열리는 포트 2(Edge -> La)

Edge_dir = os.getcwd()
read_txt = {}

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
    
#=================================#
# communication result data class #
#=================================#
class variable_data:
    def __init__(self):
        self.Good_Device_IP = []  # ex) ['127.0.0.1', '127.0.0.1', 127.0.0.1', ...]
        self.Dict_Good_Device_IP = None # ex) 성능이 좋은 device의 ip {0: '127.0.0.1'}
        self.total_device_performence = {}  # 성능 측정에 참여한 device info ex) [{'192.168.191.217': [8.778, 174.75, 123.75, 4.0, -26.0, 1000.0], '127.0.0.1': ...}]
        self.Dict_Good_Device_performence = {}  # 성능 측정에 참여한 전체 device info중 성능이 가장 좋은 device info 
        self.len_val = 0  # 총 5개의 performence info만 가지고 있겠다는 변수 정의
        self.total_count = 0  # n 번째 성능측정을 정의하기 위한 값 n

    def add_value(self, ip, performence):
        
        self.Good_Device_IP.append(ip)
        self.total_device_performence[self.len_val] = performence
        self.Dict_Good_Device_IP = {index: value for index, value in enumerate(self.Good_Device_IP)}
        
        # 성능이 가장 좋은 device의 info를 저장하기 위함.
        for j in range(self.len_val+1):  
            for i in range(2):  
                if self.Dict_Good_Device_IP[j] == self.total_device_performence[j][i][0]:
                    self.Dict_Good_Device_performence[j] = self.total_device_performence[j][i]
        
        #================ checking checking checking=====================#
        print(f"{self.total_count+1} Count, Best DEVICE performence = {self.Dict_Good_Device_performence}")
        print(f"{self.total_count+1} Count, BestDEVICE_ip = {self.Dict_Good_Device_IP}")
        print(f"{self.total_count+1} Count, Participate measureing DEVICE = {self.total_device_performence}\n")
        #=================================================================#
        self.len_val += 1
        self.total_count += 1
        
    
    def del_value(self):
        if len(self.total_device_performence) >= 5 and len(self.Good_Device_IP) >= 5:
            self.total_device_performence.clear();self.Dict_Good_Device_IP.clear();self.Dict_Good_Device_performence.clear()
            self.len_val = 0

    def return_nice_ip(self):
        best_ip_key = 0    
        while True:
            if self.Dict_Good_Device_IP is not None:
                best_ip_key = best_ip_key = list(self.Dict_Good_Device_IP)[-1]
                return self.Dict_Good_Device_IP[best_ip_key]
            
    def return_checking(self):
        
        if self.total_device_performence == {}:
            return None
        else:
            return 1
        
    def accuracy_cnt(self):
        return self.accuracy

        
#============#
# Edge class #
#============#        
class Edge(ConvNet, variable_data):
    def __init__(self, learning_agent_num, total_device_num, current_dir, target_device_os):
        super().__init__() 
        variable_data.__init__(self)
        self.learning_agent_num = learning_agent_num
        self.total_device_num = total_device_num
        self.target_device_os = target_device_os
        self.LAIP_list = []
        self.LA_socket_list = []
        self.LA_socket_list2 = []
        self.IP_list = [0]*total_device_num
        self.comm_list = [0]*total_device_num
        self.current_dir = current_dir        
        self.model = None   # model data
        self.accuracy = []  # accuracy data
        self.socket3_control_message = False
        self.server_socket2_control = True
        self.model_comm_count = 0 

        self.message_flag1 = True
        self.message_flag2 = None
        self.train_measure_flag = None
        self.train_control = None 
        self.train_model = None
        self.edge_train_flag = True

        self.train_time = 0
        self.train_time_list = []

        self.model_semaphore = threading.Semaphore(1)
        self.accuracy_semaphore = threading.Semaphore(1)

    #===============================#
    # always connecting with server #
    #===============================#
    def Server_connect_socket1(self):
        server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket1.connect((SERVER_IP, SERVER_PORT1))
        print("Server connection Success")
        print(f"learning agent num = {self.learning_agent_num}")
        self.model_comm_count = 0 
    
        print_always_open_port = 0
        while True:
            server_message = server_socket1.recv(1024).decode()
            print(f"Server response: {server_message}")
            server_socket1.sendall(b"connection success")

            if self.learning_agent_num == 0:
                if self.edge_train_flag == True:
                    if self.message_flag1 == True:
                        self.message_flag1 = None
                        message = "Send model"
                        server_socket1.send(message.encode())
                    if "server port2 open" in server_message:
                        Server_connect_thread1 = threading.Thread(target=self.Server_connect_socket2, args=(message,))
                        Server_connect_thread1.start()
                
                elif self.edge_train_flag == False:
                    if self.message_flag1 == False:
                        self.message_flag1 = None
                        message = "Receive model"
                        server_socket1.send(message.encode())
                    if "server port2 open" in server_message:
                        self.edge_train_flag = None
                        Server_connect_thread1 = threading.Thread(target=self.Server_connect_socket2, args=(message,))
                        Server_connect_thread1.start()
                
                if "server socket2 reopen" in server_message:
                    Server_connect_thread1 = threading.Thread(target=self.Server_connect_socket2, args=(message,))
                    time.sleep(3)
                    Server_connect_thread1.start() 

            elif self.learning_agent_num != 0:
                try:
                    if self.train_measure_flag == True:
                        if self.message_flag2 == True:            
                            message = "train measure"
                            server_socket1.send(message.encode())
                            self.message_flag2 = False
                        
                        if "server port2 open" in server_message:
                            Server_connect_thread1 = threading.Thread(target=self.Server_connect_socket2, args=(message,))
                            Server_connect_thread1.start()
                            self.train_measure_flag = False

                    if self.model_comm_count == 0 and len(self.train_time_list) == self.total_device_num:
                        if self.message_flag1 == True:
                            self.message_flag1 = False
                            message = "Send model"
                            server_socket1.send(message.encode())
                        if "server port2 open" in server_message:
                            Server_connect_thread1 = threading.Thread(target=self.Server_connect_socket2, args=(message,))
                            time.sleep(3)
                            Server_connect_thread1.start() 
                            self.model_comm_count += 1
            
                    if  self.model != None and self.accuracy != [] and self.server_socket2_control:  # La로 부터 받은 모델이 존재하면
                        if self.message_flag1 == False:
                            self.message_flag1 = None
                            message = "Receive model"
                            server_socket1.send(message.encode())
                        if "server port2 open" in server_message:
                            Server_connect_thread1 = threading.Thread(target=self.Server_connect_socket2, args=(message,))
                            time.sleep(3)
                            Server_connect_thread1.start() 

                    if "server socket2 reopen" in server_message:
                        Server_connect_thread1 = threading.Thread(target=self.Server_connect_socket2, args=(message,))
                        time.sleep(3)
                        Server_connect_thread1.start() 
                
                    elif self.model == None:
                        continue
                
                except socket.error as e:
                    if e.errno == 9:
                        print("소켓이 닫혀있습니다.")
                    else:
                        print("소켓 에러 발생:", e)
                    
                
                    self.model_comm_count += 1
                print_always_open_port += 1
                time.sleep(1)
    
    def Server_connect_socket2(self, message):
    
        self.server_socket2_control = False 
        
        server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket2.connect((SERVER_IP, SERVER_PORT2))
        print("server port2 open")
        message = server_socket2.recv(4)  # "send model..."
            
        if message:
            print(f"From server ={message}")

            if message == "0" or message == b"0":
                exit(1)
        
            if b'send' in message:
                self.model_semaphore.acquire()
                self.accuracy_semaphore.acquire()
                print("Model Receving start")                     
                self.model_receiving(None, server_socket2)
        
                self.model_semaphore.release()
                self.accuracy_semaphore.release()
                
                if self.learning_agent_num == 0:
                    train_thread = threading.Thread(target=self.train_measure)
                    train_thread.start()

            
            elif b'rece' in message:
                self.model_semaphore.acquire()
                self.accuracy_semaphore.acquire()

                print("Model Resending start")

                self.model_sending(None, server_socket2) 
                    
                self.model_semaphore.release()
                self.accuracy_semaphore.release()

                server_socket2.close();print("server socket2 exit")
                    
        exit(3)
        
    
    def handle1_LA(self, LA_SOCKET, LA_ADDRESS):

        k = 0

        self.LAIP_list.append(LA_ADDRESS)
        self.LA_socket_list.append(LA_SOCKET)
        print(f"CONNECT LA : {self.LAIP_list}")
 
        while True:
            LA_SOCKET.sendall(b"Using [always open] port") 
            message = LA_SOCKET.recv(1024).decode()

            if k % 10 == 0:
                print(f"{LA_SOCKET.getpeername()} : {message}")
            
            ###  opening socket2 ###
            if k % 420 == 0:
                self.model_comm_count = 0
                self.train_time_list.clear()
                self.LA_socket_list2.clear()
                self.train_control = True
                self.train_measure_flag = True
                self.message_flag1 = True
                self.message_flag2 = True
                La_thread2 = threading.Thread(target=self.La_bind_socket2, args=(k,))
                LA_SOCKET.sendall(b"Socket2 open")
                La_thread2.start()
            
            ###  opening socket3 ###
     
            if  (self.model != None and self.socket3_control_message == False) or (self.model != None and self.train_measure_flag == False):
                La_thread3 = threading.Thread(target=self.La_bind_socket3)

                if self.train_measure_flag == False:
                    for i in range(self.learning_agent_num):
                        self.LA_socket_list[i].send("measure".encode())
                        La_thread3.start()

                # 성능이 좋은 DEVICE 에게만 지정해서 보냄
                else:
                    sec = []
                    min_sec = 0
                    for i in range(len(self.train_time_list)):
                        sec.append(self.train_time_list[i][0])
                        if len(sec) == len(self.train_time_list):
                            min_sec = min(sec)
                            sec.clear()
    
                    for item in self.train_time_list:
                        if min_sec in item:
                            for i in range(self.learning_agent_num):
                                if self.LAIP_list[i][0] == item[1]:
                                    print("MODEL send to : ",self.LAIP_list[i][0])
                                    self.LA_socket_list[i].send("Socket3 open".encode())
                                    La_thread3.start()

            
            time.sleep(1)            
            k += 1
    
    def handle2_LA(self, LA_SOCKET2, LA_ADDRESS2):
        self.LA_socket_list2.append(LA_SOCKET2)

        for i in range(len(self.LA_socket_list2)):
            self.comm_measure_main(i)
            read_txt = self.read()
            str_to_float = self.str_to_float(read_txt)
            new_str_to_float = [[key] + [float(val) if isinstance(val, str) and val.replace('.', '').isdigit() else val for val in values] for key, values in str_to_float.items()]
            super().add_value(self.compare(str_to_float), new_str_to_float);super().del_value()  # variable_data 클래스에 Good Device ip 값을 append

    def La_bind_socket1(self):
        #  defineing Server_thread 
        Server_thread = threading.Thread(target=self.Server_connect_socket1)
        Server_thread.start()   
        k = 0

        Edge_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        Edge_socket1.bind((EDGE_IP, EDGE_PORT1))
        Edge_socket1.listen(self.learning_agent_num)

        print("Edge_socket1 Start...")
    
        i = 0 
        while True:
            LA_SOCKET, LA_ADDRESS = Edge_socket1.accept()
            LA_thread = threading.Thread(target=self.handle1_LA, args=(LA_SOCKET, LA_ADDRESS,))

            LA_thread.start() 
            
    def La_bind_socket2(self, k):
        Edge_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        Edge_socket2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        Edge_socket2.bind((EDGE_IP, EDGE_PORT2))
        Edge_socket2.listen(self.learning_agent_num)

        print("\nEdge_socket2 Start...")

        while True:
            LA_SOCKET2, LA_ADDRESS2 = Edge_socket2.accept()
            print(f"LA connection socket2 : {LA_SOCKET2.getpeername()}")
            LA_thread2 = threading.Thread(target=self.handle2_LA, args=(LA_SOCKET2, LA_ADDRESS2,))
            LA_thread2.start()
            break
                
    def La_bind_socket3(self):    
        if self.train_measure_flag == False:
            self.train_measure_flag = None
        Edge_socket3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        Edge_socket3.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        Edge_socket3.bind((EDGE_IP, EDGE_PORT3))
        Edge_socket3.listen(1)
        print("\nsokcet3 open...")
        LA_SOCKET3, LA_ADDRESS3 = Edge_socket3.accept()
        print(f"Socket3 bind La info : {LA_ADDRESS3}")
        self.socket3_control_message = True
        
        time.sleep(3)  # 3 sec stop for synchronization 
        
        if self.model != None:
            self.model_semaphore.acquire()
            self.model_sending(LA_SOCKET3, None)
            self.model_semaphore.release()

        self.model_semaphore.acquire()
        self.accuracy_semaphore.acquire()
        self.model_receiving(LA_SOCKET3, None)
        self.model_semaphore.release()
        self.accuracy_semaphore.release() 
        

    def model_sending(self, LA_SOCKET3, SERVER_SOCKET2): 
        if LA_SOCKET3 != None and SERVER_SOCKET2 == None:
            LA_SOCKET3.send("Model sending...".encode())
            
            if self.train_control == True and self.learning_agent_num != 0:
                self.train_control = False
                self.train_model = copy.deepcopy(self.model) 
                train_measure_thread = threading.Thread(target=self.train_measure)
                train_measure_thread.start() 

            model.load_state_dict(self.model)            
            self.model = None
            weight = model.state_dict().items()
            dict_weight = dict(weight)
            send_weight = pickle.dumps(dict_weight)
            LA_SOCKET3.send(len(send_weight).to_bytes(4, byteorder='big'))
            LA_SOCKET3.send(send_weight)  

        elif LA_SOCKET3 == None and SERVER_SOCKET2 != None:
            dict_weight = dict(self.model)
            self.model = None
            send_weight = pickle.dumps(dict_weight)

            SERVER_SOCKET2.sendall(struct.pack('>I', len(send_weight)))
            SERVER_SOCKET2.sendall(send_weight)

            if SERVER_SOCKET2.recv(1024).decode() == "accuracy":
                SERVER_SOCKET2.send(self.accuracy[0].encode())
                self.accuracy = [] 
    
    def model_receiving(self, LA_SOCKET3, SERVER_SOCKET2):  
        if LA_SOCKET3 != None and SERVER_SOCKET2 == None and self.learning_agent_num!=0:
            LA_SOCKET3.recv(1024)
            LA_SOCKET3.send('1'.encode())

            data_size = struct.unpack('>I', LA_SOCKET3.recv(4))[0]
            received_payload = b""
            remaining_payload_size = data_size
            while remaining_payload_size != 0:
                received_payload += LA_SOCKET3.recv(remaining_payload_size)
                remaining_payload_size = data_size - len(received_payload)
            self.model = pickle.loads(received_payload)
            print(f"From La = {self.model['conv1.weight'][0][0][0][0:3]}")  # checking 
            if self.train_control == False:
                self.train_control = None
                self.model = None
                LA_SOCKET3.send("send train time".encode())
                self.train_time = float(LA_SOCKET3.recv(1024).decode()) 
                LA_SOCKET3.send('1'.encode())
                LA_IP = LA_SOCKET3.recv(1024).decode()

                train_list = [self.train_time, LA_IP]

                self.train_time_list.append(train_list)

            else:
                LA_SOCKET3.send("send accuracy".encode())
                self.server_socket2_control = True
                self.message_flag1 = False 
                LA_SOCKET3.send('1'.encode())
                accuracy = LA_SOCKET3.recv(1024).decode()
                self.accuracy.append(str(accuracy))
                print(f"Accuracy = {self.accuracy[0]}%")



        elif LA_SOCKET3 == None and SERVER_SOCKET2 != None:
            data_size = struct.unpack('>I', SERVER_SOCKET2.recv(4))[0]
            received_payload = b""
            remaining_payload_size = data_size
             
            while remaining_payload_size != 0:
                received_payload += SERVER_SOCKET2.recv(remaining_payload_size)
                remaining_payload_size = data_size - len(received_payload)
            self.model = pickle.loads(received_payload)
            print(f"From server = {self.model['conv1.weight'][0][0][0][0:3]}")  # checking
            self.socket3_control_message = False



    def train_measure(self):    
        start_time = None;end_time = None;EDGE_measure_list = []
        
        if self.learning_agent_num == 0:
            self.model_semaphore.acquire()
            self.accuracy_semaphore.acquire()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(777)
        if device == 'cuda':
            torch.cuda.manual_seed_all(777)
        learning_rate = 0.001
        batch_size = 100
        num_classes = 10
        epochs = 3
        if self.learning_agent_num == 0:
            epochs = 1
            
        train_set = torchvision.datasets.MNIST(
        root = './data/MNIST',
        train = True,
        download = True,
        transform = transfroms.Compose([
            transfroms.ToTensor() 
            ]))
            
        test_set = torchvision.datasets.MNIST(
            root = './data/MNIST',
            train = False,
            download = True,
            transform = transfroms.Compose([
                transfroms.ToTensor() 
            ]))
            
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

        examples = enumerate(train_set)
        batch_idx, (example_data, example_targets) = next(examples)
        example_data.shape
        #model = ConvNet().to(device)
        
        for i in range(1):
            if self.learning_agent_num == 0:
                model.load_state_dict(self.model)
            else:
                model.load_state_dict(self.train_model)
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
            for epoch in range(epochs):
                start_time = time.time()
                avg_cost = 0
        
                for batch_idx, (data, target) in enumerate(train_loader):
                    if batch_idx % 50 == 1:  
                        data = data.to(device)
                        target = target.to(device)
                        optimizer.zero_grad()
                        hypothesis = model(data) 
                        cost = criterion(hypothesis, target)
                        cost.backward()
                        optimizer.step() 
                        avg_cost += cost / len(train_loader)
                end_time = time.time()
                self.train_time += (end_time - start_time)
            
                
                print('\n[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
        EDGE_measure_list = [self.train_time, EDGE_IP]
        self.train_time_list.append(EDGE_measure_list)
        print(f"Edge train time = {self.train_time}sec") 
        print(f"Total train_time list = {self.train_time_list}") 
        model.eval() 

        with torch.no_grad():  
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx % 50 == 1:  
                    data = data.to(device)
                    target = target.to(device)
                    out = model(data)
                    preds = torch.max(out.data, 1)[1] 
                    total += len(target) 
                    correct += (preds==target).sum().item()
            if self.learning_agent_num == 0:
                self.accuracy.append(str(100.*correct/total))

            print('Test Accuracy: ', 100.*correct/total, '%')
        
        if self.learning_agent_num == 0:
            self.model = model.state_dict().items() 
            self.edge_train_flag = False
            self.message_flag1 = False
            self.model_semaphore.release()
            self.accuracy_semaphore.release()
        

    def throughput(self, i):
        buf = 0
        while True:
            data = self.LA_socket_list2[i].recv(1024)
            buf += len(data)
            if buf >= 100000000:
                break

    def write_file(self, filename, ip, th, rtt, jitter, hop, rs, ls):
        with open(filename, 'w') as f:
            f.write(ip + ' ')
            f.write(th + ' ')
            f.write(rtt + ' ')
            f.write(jitter + ' ')
            f.write(hop + ' ')
            f.write(rs + ' ')
            f.write(ls)
        
        print(f"write_file = {ip, th, rtt, jitter, hop, rs, ls}")


    def test(self, ip, filename):
        if self.target_device_os == 'window':
            p = ['ping', ip]
            j = ['tracert', ip]
        else:
            p = ['ping', '-c', '4', ip]
            h = ['traceroute', ip]

        ji = 0
        hop_count = [] 

        f = open(filename, 'w')

        out = subprocess.Popen(p,stdout=subprocess.PIPE).stdout
        data = out.read().strip()
        out.close()

        buffers = data.decode('cp949', 'ignore')
        buff_str = str(buffers)
        buff_arr = buff_str.split("\n")

        if self.target_device_os == 'window':
            pattern = re.compile(r'\d{1, 3}ms')
            pattern2 = re.compile(r'\d\d\d\dms')
        else:
            pattern = re.compile(r'time=\d+\.\d+')
            pattern2 = re.compile(r'time=\d')

        num = 4
        sum = 0
        if len(buff_arr) == 8:
            print('100% loss')
            f.write('loss' + ' ')
        else:
            for i in range(num):
                if self.target_device_os == 'window':
                    try:
                        result = re.findall(pattern, buff_arr[i+1]) 
                        rtt = result[0].split('ms')[0]
                    except IndexError:
                        result = re.findall(pattern2, buff_arr[i+1])
                        rtt = result[0].split('ms')[0]
                else:  # Ubuntu os
                    try:
                        result = re.findall(pattern, buff_arr[i+1]) 
                        rtt = result[0].split('=')[1]
                    except IndexError:
                        result = re.findall(pattern2, buff_arr[i+1])
                        rtt = result[0].split('=')[1]

                sum += float(rtt)
                print('RTT  ' + rtt + ' ms')
            print('RTT Average ' , round((sum/num), 3) , ' ms')

            for i in range(num-1):
                if self.target_device_os == 'window':
                    try:
                        ji += abs(float((re.findall(pattern, buff_arr[i+2]))[0].split('ms')[1]) - float((re.findall(pattern, buff_arr[i+1]))[0].split('ms')[1]))
                    except IndexError:
                        ji += abs(float((re.findall(pattern2, buff_arr[i+2]))[0].split('=')[1]) - float((re.findall(pattern2, buff_arr[i+1]))[0].split('=')[1]))
                else:  # Ubuntu os
                    try:
                        ji += abs(float((re.findall(pattern, buff_arr[i+2]))[0].split('=')[1]) - float((re.findall(pattern, buff_arr[i+1]))[0].split('=')[1]))
                    except IndexError:
                        ji += abs(float((re.findall(pattern2, buff_arr[i+2]))[0].split('=')[1]) - float((re.findall(pattern2, buff_arr[i+1]))[0].split('=')[1]))
            jitter = round(ji/num, 3)
            print('Jitter : ' , jitter , 'ms')

        # Count Hop
        out = subprocess.Popen(h, stdout=subprocess.PIPE).stdout
        data = out.read().strip()
        out.close()

        buffers = data.decode('cp949', 'ignore')
        buff_str = str(buffers)
        buff_arr = buff_str.split("\n")
    
        for i in range(len(buff_arr)):
            if "*" in buff_arr[i]:
                break
            hop_count.append(buff_arr[i])

        if self.target_device_os == 'window':
            hop = len(hop_count)-3
        else:  # Ubuntu os
            hop = len(hop_count)

        if hop >= 30:
            print('More Than 30, (30 hops max...)')

        else:
            print("Hop Count :" , hop)

        return str(round((sum/num), 3)), str(jitter), str(hop)

    def throughput2(self, i):
        print('Throughput measure...')
        while True:
            b = os.path.getsize('th.txt')
            w = open('th.txt', 'rb')
            j = w.read(b) 

            t1 = time.time()
            while(j):
                self.LA_socket_list2[i].send(j)
                j = w.read(b)
            t2 = time.time()
            break

        th = round((b*0.001*0.001) / (t2-t1), 3)
        print(f'Throughtput: {th} MB/sec')

        return str(th)

    # Link Speed, RSSI 측정하는 함수
    def measure2(self):
        if self.target_device_os == 'window':
            l = r = ['netsh', 'wlan', 'show', 'interfaces']
        else:  # Ubuntu os
            l = ['ip', 'addr']
            r = ['iw', 'dev', 'wlan0', 'link']
        print('Link Speed/RSSI measure...')

        # Link Speed
        out = subprocess.Popen(l, stdout=subprocess.PIPE).stdout
        data = out.read().strip()
        out.close()

        buffers = data.decode('cp949', 'ignore')
        buff_str = str(buffers)

        if self.target_device_os == 'window':
            pattern = re.compile(r'수신 속도\(Mbps\) +: (\d+\.+\d)')
            link_speed = re.findall(pattern, buff_str)
            if link_speed == []:
                pattern = re.compile(r'수신 속도\(Mbps\) +: (\d+)')
                link_speed = re.findall(pattern, buff_str)
            link_speed = link_speed[0]
        else:  # opt.target_device_os == 'Ubuntu'
            pattern = re.compile(r'qlen\s\d\d\d\d')
            result = re.findall(pattern, buff_str)
            link_speed = result[1].split(' ')[1]

        print("Link Speed :", link_speed, "Mb/s")

        # RSSI
        out = subprocess.Popen(r, stdout=subprocess.PIPE).stdout
        data = out.read().strip()
        out.close()

        buffers = data.decode('cp949', 'ignore')
        buff_str = str(buffers)

        if self.target_device_os == 'window':
            pattern = re.compile(r'신호 +: (\d+)%')
            signal_strength = re.findall(pattern, buff_str)
            rssi_percentage = int(signal_strength[0])
            rssi = (rssi_percentage/2) - 100

        else:  # opt.target_device_os == 'Ubuntu'
            pattern = re.compile(r'signal: \S\d\d')
            result = re.findall(pattern, buff_str)
            rssi = result[0].split(': ')[1]

        print("RSSI (Signal level) :", rssi, "dBm")
        return str(link_speed), str(rssi)
    
    def comm_measure_main(self, i): 
        r = self.LA_socket_list2[i].recv(1024).decode() 
        self.LA_socket_list2[i].send('OK'.encode())
        ip = self.LA_socket_list2[i].recv(1024).decode()
        la_id = int(r)
    
        filename = 'La' + str(la_id+1) + '.txt'

        self.throughput(i)
        self.LA_socket_list2[i].send('1'.encode())
        th = self.LA_socket_list2[i].recv(1024).decode()
        self.LA_socket_list2[i].send('1'.encode())
        ls = self.LA_socket_list2[i].recv(1024).decode()
        self.LA_socket_list2[i].send('1'.encode())
        rs = self.LA_socket_list2[i].recv(1024).decode()
            
        print(f"throughput = {th}")
        print(f"linkSpeed = {ls}")
        print(f"RSSI = {rs}")

        rtt, jitter, hop = self.test(ip, filename)

        self.write_file(filename, ip, th, rtt, jitter, hop, rs, ls) 
            
        # measure2 START
        self.LA_socket_list2[i].send("measure2".encode())  # measure2 ->
        self.LA_socket_list2[i].recv(1024)  # ok <- 

        self.LA_socket_list2[i].send(EDGE_IP.encode())  # Device IP -> 


        th2 = self.throughput2(i)  # Throughput
    
        ls2, rs2 = self.measure2()
        
        rtt2 = self.LA_socket_list2[i].recv(1024).decode()
        self.LA_socket_list2[i].send('1'.encode())
        jitter2 = self.LA_socket_list2[i].recv(1024).decode()
        self.LA_socket_list2[i].send('1'.encode())
        hop2 = self.LA_socket_list2[i].recv(1024).decode()
        self.LA_socket_list2[i].send('1'.encode())

        print(f"th2 = {th2}\nls2 = {ls2}\nrs2 = {rs2}\nrtt2 ={rtt2}\njitter2 = {jitter2}\nhop2 = {hop2}")
        self.write_file('Edge.txt', EDGE_IP, th2, rs2, jitter2, hop2, rs2, ls2)
        self.LA_socket_list2[i].recv(1024) # ok <- i
      
                
    """===================Learning Agent로 부터 성능측정.txt정보 수신 메서드================"""
    def receive_file_from_la(self):

        print("Receive start...")

        for i in range(self.learning_agent_num):
            with  open(self.current_dir + f"/La{i+1}.txt", "wb") as file:
                data = self.LA_SOCKET2[i].recv(1024)
                #print(f"{data}")
                file.write(data)
                if not data:
                    break
        print("파일이 성공적으로 수신되었습니다.")
    
    """===================저장된 성능측정 정보(.txt파일) 읽어오는 메서드==================="""
    def read(self): 
        print("Read...")
        for j in range(self.total_device_num):  
            if j == 0:   # i==0이면 Edge의 성능측정 정보를 read하여 저장
                filename = Edge_dir + '/Edge.txt'
                with open(filename, 'r') as file:
                    data = file.read()
                    s_data = str(data)
                    self.IP_list[0] = s_data.split(' ')[0]
                    self.comm_list[0] = [item.strip() for item in s_data.split(' ')[1:]]
                    #  strip() 함수는 문자열의 양쪽 끝에 있는 공백 문자와 개행문자 등을 제거
            else:   # i > 0이면 Learning agent의 성능측정 정보를 read하여 저장
                filename = self.current_dir + '/La' + str(j) + '.txt'
                with open(filename, 'r') as file:
                    data = file.read()
                    s_data = str(data)
                    self.IP_list[j] = s_data.split(' ')[0]
                    self.comm_list[j] = [item.strip() for item in s_data.split(' ')[1:]]
        
        for k in range(self.total_device_num):
            read_txt[self.IP_list[k]] = self.comm_list[k]
        
        return read_txt
    
    """=============리스트 내의 문자열 데이터를 float 데이터로 변환후 반환 메서드=================== """
    def str_to_float(self, network_string):
        txt_to_float = network_string
        txt_to_float = {ip : [float(comm_data) for comm_data in comm_datas] for ip, comm_datas in txt_to_float.items()}
        return txt_to_float

    """================== 최소값 비교 메서드================="""
    def compare(self, network_float):
        print("compare\n")
        a = network_float
        comm = np.array([a[i] for i in a])
        best_index = np.where(comm[:, 0] == np.max(comm[:, 0]))[0]  ## Throughput
        IP = [list(a.keys())[i] for i in best_index]

        if len(IP) == 1:
            return IP[0]
        for i in list(a.keys()):
            if i not in IP:
                del a[i]

        if (len(IP) >= 2):
            comm = np.array([a[i] for i in a])
            best_index = np.where(comm[:, 1] == np.min(comm[:, 1]))[0]  ## Rtt
            IP = [list(a.keys())[i] for i in best_index]
            if len(IP) == 1:
                return IP[0]
            for i in list(a.keys()):
                if i not in IP:
                    del a[i]

            if (len(IP) >= 2):
                comm = np.array([a[i] for i in a])
                best_index = np.where(comm[:, 2] == np.max(comm[:, 2]))[0]  ## jitter
                IP = [list(a.keys())[i] for i in best_index]
                if len(IP) == 1:
                    return IP[0]
                for i in list(a.keys()):
                    if i not in IP:
                        del a[i]

                if (len(IP) >= 2):
                    comm = np.array([a[i] for i in a])
                    best_index = np.where(comm[:, 3] == np.min(comm[:, 3]))[0]  ## Hop count
                    IP = [list(a.keys())[i] for i in best_index]
                    if len(IP) == 1:
                        return IP[0]
                    for i in list(a.keys()):
                        if i not in IP:
                            del a[i]

                    if (len(IP) >= 2):
                        comm = np.array([a[i] for i in a])
                        best_index = np.where(comm[:, 4] == np.min(comm[:, 4]))[0]  ## Rssi
                        IP = [list(a.keys())[i] for i in best_index]
                        if len(IP) == 1:
                            return IP[0]
                        for i in list(a.keys()):
                            if i not in IP:
                                del a[i]

                        if (len(IP) >= 2):
                            comm = np.array([a[i] for i in a])
                            best_index = np.where(comm[:, 3] == np.min(comm[:, 3]))[0]  ## LinkSpeed
                            IP = [list(a.keys())[i] for i in best_index]
                            if len(IP) == 1:
                                return IP[0]

    

    
