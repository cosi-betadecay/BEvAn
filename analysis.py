import ROOT as M
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),     
        logging.FileHandler("logs/analysis.log")
    ]
)

def read_file(filename: str):
    with open(filename, "r") as f:
        for line_number, line in enumerate(f, start=1):
            logging.info(f"Line {line_number}: {line.strip()}")