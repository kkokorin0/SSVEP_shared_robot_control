import socket


class StimController:
    """Control SSVEP stimuli presented in HoloLens via TCP/IP"""

    bufsize = 1024  # buffer size
    port = 25001  # port number

    def __init__(self, host, logger):
        """Connect to HoloLens via TCP/IP

        Args:
            host (str): host IP
            logger (logging): logger object
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, self.port))
        self.logger = logger

    def send_data(self, msg, get_response=True):
        """Send data to HoloLens

        Args:
            msg (str): message to send
            get_response (bool, optional): whether to wait for response. Defaults to True.
        """
        try:
            # connect to server and send msg
            self.socket.sendall(msg.encode("utf-8"))

            if get_response:
                response = self.socket.recv(self.bufsize).decode("utf-8")
                self.logger.debug("Response:" + response)

        except Exception as e:
            self.logger.critical(f"An error occurred when sending message: {e}")

    def setup_stim(self, freqs):
        """Reset stimulus to initial state

        Args:
            freqs (np.array): array of stimulus frequencies
        """
        self.send_data("reset:%s" % ",".join("%.3f" % f for f in freqs))

    def move_stim(self, ef_coords):
        """Move stimuli to match end-effector coordinates

        Args:
            ef_coords (np.array): [x,y,z] coordinates of end-effector
        """
        # Neet to update to match HoloLens coordinate system
        self.send_data(
            "move:%.3f,%.3f,%.3f" % (ef_coords[0], ef_coords[1], ef_coords[2])
        )

    def end_run(self):
        """Turn off the app"""
        self.setup_stim([0, 0, 0, 0, 0])
        self.send_data("quit: ", get_response=False)
