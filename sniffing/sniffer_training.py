class SnifferTraining:
    def __init__(self, target_app):
        self.target_app = target_app
    def start(self):
        print("There will be a script using psutil, which will save a pcap file per process.")