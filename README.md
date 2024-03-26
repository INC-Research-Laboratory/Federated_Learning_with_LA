# Federated_Learning_with_LA

- Dataset : MNIST
- Model : Custom CNN

연합학습으로 MNIST dataset을 사용하여 성능이 더 좋은 기기(Edge or LearningAgent)에서 학습을 진행한다.
# Device
 - Desktop : Server
 - Jetson XavierNX : Client 1, Client 2
 - Asus Laptop : LearningAgent 1
 - Jetson AGX ORIN : LearningAgent 2


## run
```bash
python Server_main.py  # runining on the Desktop
python Edge_main.py  # runining on the XavierNX
python La_main.py  # runining on the Asus Laptop, ORIN
```

## Server_main side
- strategy
  - edge_device_num=2
    - 학습에 참여하는 Edge device 수

## Edge_main side
- strategy
  - total_device_num=2
    - Edge 자기 자신과 LearningAgent의 총 개수
  - learning_agent_num=1
    - LearningAgent 수
  - time_slice=20
    - LearningAgent와 몇 분에 한 번씩 성능측정 할 지에 대한 주기적 시간
  - target_device_os=window
    - 성능측정 대상이 되는 device의 OS

## La_main side
- strategy
  - target_device_os=window
    - 성능측정 대상이 되는 device의 OS
