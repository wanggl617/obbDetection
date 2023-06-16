for i in range(1, 3066):
    print("\r", end="")
    print("{}".format(i), ">" * (i // 2), end=" 14.1 task/s, elapsed: 88s, ETA:   130s")
    sys.stdout.flush()
    time.sleep(0.05)
