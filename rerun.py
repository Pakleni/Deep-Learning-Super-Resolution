while True:
    try:
        exec(open("./main.py").read())
    except SystemExit:
        pass
    try:
        exec(open("./mainGAN.py").read())
    except SystemExit:
        pass