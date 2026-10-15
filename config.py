from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'original' / 'twitter_disaster.xlsx'
ARTIFACTS = ROOT / 'artifacts'
DISCLAIMER = 'This application predicts whether text resembles a disaster-related tweet. Predictions are model outputs, not verified emergency reports. Results require human verification before operational decisions.'
