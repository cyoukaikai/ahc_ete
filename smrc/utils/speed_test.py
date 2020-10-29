import time


def benchmark(method1, method2, **kwargs):
    start_time = time.time()
    result1 = method1(
        **kwargs
    )
    elapsed_time1 = time.time() - start_time
    print(f'Method1: elapsed_time = {elapsed_time1} ... ')

    start_time = time.time()
    result2 = method2(
        **kwargs
    )
    elapsed_time2 = time.time() - start_time
    print(f'Method2: elapsed_time = {elapsed_time2} ... ')

    print(f'Method2 : {elapsed_time2} \n'
          f' ({elapsed_time1 / elapsed_time2}) times faster ... ')
    print(f'Equal or not: {result1 == result2}')
