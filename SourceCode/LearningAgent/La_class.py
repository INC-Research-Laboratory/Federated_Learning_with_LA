import socket
import os
import threading
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transfroms
import torchvision
import copy
import torch
import pickle
import struct
from collections import OrderedDict
import numpy as np 
import time
import subprocess
import re 

EDGE_IP = "Edge IP"
EDGE_PORT1 = 20001 # 항상 열려있는 포트(Edge -> La)
EDGE_PORT2 = 30001  # 주기적으로 열리는 포트 1(Edge -> La)
EDGE_PORT3 = 35001     # 주기적으로 열리는 포트 2(Edge -> La)

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
    
class LearningAgent(ConvNet):
    def __init__(self, current_dir, target_device_os):
        super().__init__()
        self.current_dir = current_dir
        self.model = None
        self.accuracy = None
        self.target_device_os = target_device_os
        self.this_ip = 'La ip'

        self.model_semaphore = threading.Semaphore(1)
        self.accuracy_semaphore = threading.Semaphore(1)

        self.measure_train = None
        self.train_time = 0

    def Connect_Edge1(self):
        
        La_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        La_socket1.connect((EDGE_IP, EDGE_PORT1))
        print("Edge connect succes...")
        
        while True:
            To_Edge_message = "La connect"
            La_socket1.send(To_Edge_message.encode()) 
            
            From_Edge_message = La_socket1.recv(1024).decode()
            print("Edge message =", From_Edge_message) 

            if From_Edge_message == "Socket2 open":
                thread2 = threading.Thread(target=self.Connect_Edge2, args=(From_Edge_message,))
                thread2.start()
            
            
            elif From_Edge_message == "Socket3 open" or From_Edge_message == "measure":
                if From_Edge_message == "measure":
                    self.measure_train = True
                thread3 = threading.Thread(target=self.Connect_Edge3)
                thread3.start()
            
    def Connect_Edge2(self, message):
        try:
            La_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            La_socket2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            La_socket2.connect((EDGE_IP, EDGE_PORT2))
            print(f"\nmessage = {message}")
            self.comm_measure_main(La_socket2)
            La_socket2.send("sending".encode())
        except ConnectionRefusedError:
            print("서버에 연결할 수 없습니다. 재시도 중...")
            time.sleep(1)
    
    # Throuhgput 측정하는 함수, 100,000,000 bytes 크기 'th.txt' 파일 생성 필수
    def throughput(self, La_socket2):

        print('Throughput measure...')
        while True:
            b = os.path.getsize('th.txt')  
            w = open('th.txt', 'rb')
            j = w.read(b)

            t1 = time.time()
            while(j):
                La_socket2.send(j)
                j = w.read(b)
            t2 = time.time()
            break

        th = round((b*0.001*0.001) / (t2-t1), 3)
        print('Throughput: ', th, 'MB/sec')
        
        return str(th)
    

    # Link Speed, RSSI 측정하는 메서드
    def measure(self):
        if self.target_device_os=="window":
            l = r = ['netsh', 'wlan', 'show', 'interfaces']
        else: 
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
        else:  # Ubuntu os command

            pattern = re.compile(r'qlen\s\d\d\d\d')
            result = re.findall(pattern, buff_str)
            link_speed = result[1].split(' ')[1][0]

        print(f"Link Speed : {link_speed} Mb/s")
        


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
        else: # Ubuntu os command
            pattern = re.compile(r'signal: \S\d\d')
            result = re.findall(pattern, buff_str)
            rssi = result[0].split(': ')[1]


        print(f"RSSI (Signal level) : {rssi} dBm")
        return str(link_speed), str(rssi)
    
    def throughput2(self, La_socket2):
        buf = 0

        while True:
            data = La_socket2.recv(1024)
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
        f.close()
        print(f"write_file = {ip, th, rtt, jitter, hop, rs, ls}")



    # RTT, Jitter, Hop Count 측정하는 메서드
    def test2(self, ip, filename):
        if self.target_device_os == 'window':
            p = ['ping', ip] 
            h = ['tracert', ip] 
        else:  # opt.target_device_os == 'Ubuntu':
            p = ['ping', '-c', '4', ip] 
            h = ['traceroute', ip]
        
        ji=0
        hop_count = []

        f = open(filename, 'w')

        # Rtt & Jitter
        out = subprocess.Popen(p, stdout=subprocess.PIPE).stdout
        data = out.read().strip()
        out.close()

        buffers = data.decode('cp949', 'ignore')
        buff_str = str(buffers)
        buff_arr = buff_str.split("\n")
        
        if self.target_device_os == 'window':
            pattern = re.compile(r'\d{1,3}ms')
            pattern2 = re.compile(r'\d\d\d\dms')
        else:  # opt.target_device_os == 'Ubuntu' 
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
                else:  # opt.target_device_os == 'Ubuntu'
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
                        ji += abs(float((re.findall(pattern, buff_arr[i+2]))[0].split('ms')[0]) - float((re.findall(pattern, buff_arr[i+1]))[0].split('ms')[0]))
                    except IndexError:
                        ji += abs(float((re.findall(pattern2, buff_arr[i+2]))[0].split('ms')[0]) - float((re.findall(pattern2, buff_arr[i+1]))[0].split('ms')[0]))
                else:  #opt.target_device_os == 'Ubuntu'
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
            if '*' in buff_arr[i]:
                break
            hop_count.append(buff_arr[i])

        if self.target_device_os == 'window':
            hop = len(hop_count)-4
        else: #opt.target_device_os == 'Ubuntu':
            hop = len(hop_count)-1
            
        
        if hop >= 30:
            print('More Than 30, (30 hops max...)')

        else:
            print("Hop Count :" , hop)

        return str(round((sum/num), 3)), str(jitter), str(hop)


    def comm_measure_main(self, La_socket2):
        self.this_ip = 'La device ip'  
        ID = '0'

        La_socket2.send(ID.encode())  # id
        La_socket2.recv(1024) 
        La_socket2.send(self.this_ip.encode()) 

        th = self.throughput(La_socket2)
        ls, rs = self.measure() 

        La_socket2.recv(1024)
        La_socket2.send(th.encode())
        La_socket2.recv(1024)
        La_socket2.send(ls.encode())
        La_socket2.recv(1024)
        La_socket2.send(rs.encode())

        if La_socket2.recv(1024).decode() == "measure2":
            La_socket2.send("ok".encode())
            ip2 = La_socket2.recv(1024)
            ip2 = ip2.decode()

            filename2 = 'Edge.txt'  

            self.throughput2(La_socket2) 

            rtt2, jitter2, hop2 = self.test2(ip2, filename2)
        
            La_socket2.send(rtt2.encode())
            La_socket2.recv(1024)
            La_socket2.send(jitter2.encode())
            La_socket2.recv(1024)
            La_socket2.send(hop2.encode())
            La_socket2.recv(1024)
            La_socket2.send("ok".encode())
    
    def send_file_to_Edge(self, La_socket2):
            # Read file
            with open('Edge.txt', 'rb') as file:
                file_data = file.read()

            La_socket2.sendall(file_data)

            print("File transmission successful.\n")

    def Connect_Edge3(self):
        La_socket3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        La_socket3.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        La_socket3.connect((EDGE_IP, EDGE_PORT3))

        if La_socket3.recv(1024).decode() == "Model sending...":
            self.train(La_socket3)

    def received_model(self, La_socket3):
        start_time = None;end_time = None
        self.model_semaphore.acquire()
        self.accuracy_semaphore.acquire()
        
        if self.measure_train == True:
            start_time = time.time()     

        data_size = struct.unpack('>I', La_socket3.recv(4))[0]
        rec_payload = b""
        remaining_payload = data_size
        while remaining_payload != 0:
            rec_payload += La_socket3.recv(remaining_payload)
            remaining_payload = data_size - len(rec_payload)
        
        if self.measure_train == True:
            end_time = time.time() 
            print("\nModel receiving time = ", end_time - start_time, "sec")
            self.train_time += end_time - start_time
        dict_weight = pickle.loads(rec_payload)
        self.model = OrderedDict(dict_weight)
        self.model_semaphore.release()
        self.accuracy_semaphore.release()

    def train(self, La_socket3):
        start_time = None;end_time = None
        self.received_model(La_socket3)
        if self.model != None:
            self.model_semaphore.acquire()
            self.accuracy_semaphore.acquire()
    
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            torch.manual_seed(777)
            if device == 'cuda':
                torch.cuda.manual_seed_all(777)
            learning_rate = 0.001
            batch_size = 100
        
            epochs = 1
            if self.measure_train == True:
                epochs = 3
            
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
            print(f"TRAIN BEFORE = {self.model['conv1.weight'][0][0][0][0:3]}\n")
            for i in range(1):
            
                model.load_state_dict(self.model)
                criterion = nn.CrossEntropyLoss().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                for epoch in range(epochs):
                    avg_cost = 0
                    if self.measure_train == True:
                        start_time = time.time()
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
                    if self.measure_train == True:
                       end_time = time.time()
                       self.train_time += (end_time - start_time)
                       print(f"Train time = {self.train_time:.5f}sec")
                    
                    print('[Epoch: {:>4}] cost = {:>.9}\n'.format(epoch + 1, avg_cost))

     
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
                    print('Test Accuracy: ', 100.*correct/total, '%')
                    self.accuracy = 100.*correct/total
            self.model_semaphore.release()
            self.accuracy_semaphore.release()
            
            self.sending_model_accuracy(La_socket3)
            
            self.model_semaphore.release()
            self.accuracy_semaphore.release()
            
    def sending_model_accuracy(self, La_socket3):
        start_time = None;end_time = None

        # 학습한 모델 Edge로 전송
        print("after train send model...")
        self.model_semaphore.acquire()
        self.accuracy_semaphore.acquire()
        La_socket3.send("1".encode())
        La_socket3.recv(1024)
        self.model = model.state_dict().items()
        dict_weight = dict(self.model)
        send_weight = pickle.dumps(dict_weight)
        
        if self.measure_train == True:
            start_time = time.time()
        La_socket3.sendall(struct.pack('>I', len(send_weight)))
        La_socket3.sendall(send_weight)

        if self.measure_train == True:
            self.measure_train = None
            end_time = time.time()
            print("Model sending time = ", end_time - start_time, "sec") 
            self.train_time += (end_time - start_time)
            print("total time = ", self.train_time, "sec")
            La_socket3.send(str(self.train_time).encode())
            La_socket3.recv(1024)
            La_socket3.send(self.this_ip.encode())
            self.train_time = 0
        
        else:
            La_socket3.recv(1024)
            print("after train send accuracy...\n")
            La_socket3.send(str(self.accuracy).encode())
            self.model_semaphore.release()
            self.accuracy_semaphore.release()
        print(f"TRAIN AFTER = {dict_weight['conv1.weight'][0][0][0][0:3]}\n")
    
