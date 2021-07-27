from multiprocessing import Lock


class Counter:
    __instance = None
    __count = None
    __lock = None

    def __init__(self):
        """ Virtually private constructor. """
        if Counter.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Counter.__instance = self
            self.__count = 0
            self.__lock = Lock()

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Counter.__instance is None:
            Counter()
        return Counter.__instance

    def count(self):
        self.__lock.acquire()
        self.__count += 1
        self.__lock.release()

    def get_count(self):
        self.__lock.acquire()
        count = self.__count
        self.__lock.release()
        return count

    def reset_count(self):
        self.__lock.acquire()
        self.__count = 0
        self.__lock.release()
