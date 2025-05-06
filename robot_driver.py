import socket
import json
import random

def main():
    host = 'localhost'
    port = 5000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        while True:
            s.listen(1)
            conn, addr = s.accept()
            print("Connection established")
            with conn:
                input_stream = conn.makefile('r')
                output_stream = conn.makefile('w')
                while True:
                    random_action = random.randint(0, 3)
                    output_stream.write(str(random_action) + "\n")
                    output_stream.flush()
                    observation_json = input_stream.readline().strip()
                    if not observation_json:
                        break
                    observation = json.loads(observation_json)
                    print(observation)

if __name__ == '__main__':
    main()
