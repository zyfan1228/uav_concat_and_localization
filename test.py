import time



class Timer:
    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start = time.time()
            ret = func(*args, **kwargs)
            print(f"{self.prefix}{time.time() - start}")
            return ret
        return wrapper
    

@Timer(prefix="nihao: ")
def pow_(a, b):
    return a ** b
# 等价于 pow_ = Timer(prefix="nihao: ")(pow)

print(pow_(2, 8))
