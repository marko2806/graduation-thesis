from sku110k_dataset import SKU110kDataset

dataset = SKU110kDataset("../datasets/SKU110K", None, "test", False)

small_object_count = 0
medium_object_count = 0
large_object_count = 0

print(len(dataset))
for i in range(len(dataset)):
    if i % 100 == 0:
        print(i)
    image, label = dataset[i]
    label = label["area"].cpu().numpy()

    for j in range(len(label)):
        if label[j] < 32*32:
            small_object_count += 1
        elif label[j] < 96*96:
            medium_object_count += 1
        else:
            large_object_count += 1

print("Small objects:", small_object_count)
print("Medium objects:", medium_object_count)
print("Large objects:", large_object_count)

# Val
# Small objects: 25
# Medium objects: 21052
# Large objects: 69379
# Test
# Small objects: 0
# Medium objects: 0
# Large objects: 0
