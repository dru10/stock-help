from dataset import create_dataset

symbols = ["^SPX", "^DAX", "^BET"]

for symbol in symbols:
    lagged, cols = create_dataset(symbol)
