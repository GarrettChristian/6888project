
import threading
import sys
import time


def main():
    try:
        event = threading.Event()
        for i in range(3):
            thread = threading.Thread(target=f, args=(event, i,))
            thread.start()
        event.wait()  # wait forever but without blocking KeyboardInterrupt exceptions
    except KeyboardInterrupt:
        print("Ctrl+C pressed...")
        event.set()  # inform the child thread that it should exit
        print("Waiting for other processes to conclude then collecting results")
        sys.exit(1)

def f(event, threadId):
    while not event.is_set():
        pass  # do the actual work
        print("here %d" % (threadId))
        time.sleep(20)

if __name__ == '__main__':
    main()
