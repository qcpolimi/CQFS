from multiprocessing import cpu_count, Pool


def parallelize_function(func, args, count_div=2, count_sub=1):
    n_processes = cpu_count() // count_div - count_sub
    pool = Pool(processes=n_processes, maxtasksperchild=1)

    pool.starmap(func, args)
    pool.close()
    pool.join()
