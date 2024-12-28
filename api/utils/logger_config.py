# api/utils/logger_config.py
import logging

def setup_logging():
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logger for our app
    logger = logging.getLogger('api')
    logger.setLevel(logging.INFO)
    
    # Ensure all child loggers inherit settings
    logger.propagate = True
    
    return logger

# Create and export logger
logger = setup_logging()