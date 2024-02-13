import socket


class StimController:
    """Control SSVEP stimuli presented in HoloLens via TCP/IP"""

    bufsize = 1024  # buffer size
    port = 25001  # port number

    def __init__(self, host, logger, offset, freqs, dist):
        """Connect to HoloLens via TCP/IP

        Args:
            host (str): host IP
            logger (logging): logger object
            offset (float): coordinate frame offset [x,y,z] in robot frame
            freqs (list of float): stimuli frequencies
            dist (float): distance of outer stimuli to center (m)
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, self.port))
        self.logger = logger
        self.offset = offset
        self.freqs = freqs
        self.dist = dist

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

    def setup_stim(self, freqs, pos, dist):
        """Reset stimuli frequencies and position

        Args:
            freqs (list): stimulus frequencies
            pos (list): [x,y,z] gripper coordinates
            dist (float): distance (m) of outer stimuli from end-effector
        """
        self.send_data("setup:%.3f" % dist)
        self.move_stim(pos)
        self.send_data("reset:%s" % ",".join("%.3f" % f for f in freqs))

    def move_stim(self, ef_coords):
        """Move stimuli to match end-effector coordinates

        Args:
            ef_coords (np.array): [x,y,z] coordinates of end-effector
        """
        self.send_data(
            "move:%.3f,%.3f,%.3f" % (ef_coords[0], ef_coords[1], ef_coords[2])
        )

    def end_run(self):
        """Turn off the app"""
        self.send_data("quit: ", get_response=False)

    def turn_off_stim(self):
        """Turn off stimuli"""
        self.setup_stim([0 for _f in self.freqs], [0, 0, 0], 0)

    def turn_on_stim(self, pose):
        """Flash stimuli around end-effector

        Args:
            pose (np.array): [R11 R12 R13 Tx
                              R21 R22 R23 Ty
                              R31 R32 R33 Tz
                              0   0   0   1]
        """
        self.setup_stim(self.freqs, self.coord_transform(pose), self.dist)

    def coord_transform(self, pose):
        """Transform from Reachy to HoloLens frame

        Args:
            pose (np.array): [R11 R12 R13 Tx
                              R21 R22 R23 Ty
                              R31 R32 R33 Tz
                              0   0   0   1]

        Returns:
            list: [x,y,z] coordinates in HoloLens frame
        """
        return [
            -pose[1, 3] - self.offset[1],
            -pose[0, 3] - self.offset[0],
            pose[2, 3] + self.offset[2],
        ]

    def prompt(self, stim_states, pose):
        """Toggle stimuli at given index on/off

        Args:
            stim_states (list of bool): stimulus on or off
            pose (np.array): [R11 R12 R13 Tx
                              R21 R22 R23 Ty
                              R31 R32 R33 Tz
                              0   0   0   1]
        """
        self.setup_stim(stim_states, self.coord_transform(pose), self.dist)
