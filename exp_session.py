# Imports
import socket
import time


class StimController:
    bufsize = 1024
    port = 25001
    host = "10.15.254.106"  # HoloLens
    # host = "127.0.0.1"  # UnitySim

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        pass

    def send_data(self, msg, get_response=True):
        try:
            # connect to server and send msg
            self.socket.sendall(msg.encode("utf-8"))

            if get_response:
                response = self.socket.recv(self.bufsize).decode("utf-8")
                print("Response:", response)

        except Exception as e:
            print(f"An error occurred: {e}")

    def setup_stim(self, freqs):
        self.send_data("reset:%s" % ",".join("%.3f" % f for f in freqs))

    def move_stim(self, direction, step_size):
        self.send_data("move:%s,%.3f" % (direction, step_size))

    def end_run(self):
        self.setup_stim([0, 0, 0, 0])
        self.socket.close()


# Constants
N_TRIALS = 10
ROBOT_VEL = 0.05  # m/s
SAMPLE_T = 0.2  # s
STEP_SIZE = ROBOT_VEL * SAMPLE_T  # m
FREQS = [14, 15, 17, 18]  # grab something from literature

if __name__ == "__main__":
    unity_game = StimController()

    for trial in range(N_TRIALS):
        print("Running trial %d/%d" % (trial + 1, N_TRIALS))
        unity_game.setup_stim(FREQS)  # reset the stimuli
        # exit(0)

        run_trial = True
        while run_trial:
            # get user command
            direction = "d"
            # move the robot
            # update stimuli
            unity_game.move_stim(direction, STEP_SIZE)
            time.sleep(SAMPLE_T)  # change to use pygame clock

    unity_game.end_run()
    # reset robot
