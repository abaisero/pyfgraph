import logging

def setup_default_logger():
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logging.basicConfig(
        filename='log.log',
        filemode='w',
        format=fmt,
        level=logging.DEBUG
    )

    logger = logging.getLogger()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

