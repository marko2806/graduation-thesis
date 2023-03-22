
def evaluate(model, data_set, device):
    for image_batch, label_batch in iter(data_set):
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        out = model(image_batch)

    return None