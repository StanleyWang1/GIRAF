import winpcapy  # if installed
import pysoem

for dev in pysoem.find_adapters():
    print(dev)
