import logging

fmt = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_file_logger(fname = 'log.log', level = logging.DEBUG):
    logging.basicConfig(filename=fname,
                        filemode='w',
                        format=fmt,
                        level=level)

def setup_stream_logger(level = logging.INFO):
    logger = logging.getLogger()

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

