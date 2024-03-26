import os
from La_class import LearningAgent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target_device_os", type=str, default="window", help="target_device_os")   # 몇 분에 한 번 성능측정을 할 지
opt = parser.parse_args()

current_dir = os.getcwd() + '/La1.txt'

if __name__ == '__main__':
    La = LearningAgent(current_dir, opt.target_device_os)
    La.Connect_Edge1()
