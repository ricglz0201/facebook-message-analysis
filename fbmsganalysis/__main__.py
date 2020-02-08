import argparse
from analyzer import analyze

parser = argparse.ArgumentParser(description='Tool to analyze your Facebook Messenger history')
parser.add_argument('file', help='Facebook chat messages in JSON format')

args = parser.parse_args()
analyze(args.file)
