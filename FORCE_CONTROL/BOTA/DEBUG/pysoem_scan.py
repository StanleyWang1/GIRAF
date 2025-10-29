import pysoem
import time

m = pysoem.Master()
m.open('enp45s0')

if m.config_init() > 0:
    s = m.slaves[0]
    print(s.name)
    print(hex(s.man))
    print(hex(s.id))
    print(hex(s.rev))
    print(hex(s.serno))
else:
    print("No slaves found")

time.sleep(1)
m.close()