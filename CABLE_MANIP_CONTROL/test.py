import numpy as np

# hello = np.ones(3, dtype=np.float32)
# hello2 = hello
# hello *= 2

# hello = np.ones(3, dtype=np.float32)
# hello2 = hello.copy()
# hello *= 2

hello = np.ones(3, dtype=np.float32)
hello2 = hello
hello = hello * 2

print(hello)
print(hello2)  # This will still print the original values since hello2 is a