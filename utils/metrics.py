def pairwise_accuracy(preds, labels):
    correct = (preds == labels).sum()
    return correct / len(labels)
