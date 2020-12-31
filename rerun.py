while True:
    try:
        exec(open("./main.py").read())
    except SystemExit:
        pass