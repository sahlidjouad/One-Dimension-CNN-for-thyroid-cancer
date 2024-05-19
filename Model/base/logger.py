import logging
import os

class LoggerManager:
    def __init__(self, fileName):
        
        self.fileName = fileName
        logging.basicConfig(filename=fileName, format='%(asctime)s %(message)s',  filemode='a') 
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO) 


    def log_info(self, message):
        self.logger.info(message)