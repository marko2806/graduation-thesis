
from sku110k_dataset import SKU110kDataset
import argparse


def count_objects_by_size(dataset):
    small_object_count = 0
    medium_object_count = 0
    large_object_count = 0

    print(len(dataset))
    for i in range(len(dataset)):
        if i % 100 == 0:
            print(i)
        _, label = dataset[i]
        label = label["area"].cpu().numpy()

        for j in range(len(label)):
            if label[j] < 32*32:
                small_object_count += 1
            elif label[j] < 96*96:
                medium_object_count += 1
            else:
                large_object_count += 1
    return small_object_count, medium_object_count, large_object_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str,
                        default="./dataset/SKU110K")
    parser.add_argument('--mode', type=str, default="test", required=False)
    args, _ = parser.parse_known_args()

    dataset = SKU110kDataset(args.dataset_path, None, args.mode, False)

    small_object_count, medium_object_count, large_object_count = count_objects_by_size(
        dataset)

    print("Small objects:", small_object_count)
    print("Medium objects:", medium_object_count)
    print("Large objects:", large_object_count)

# Output:
# Val
# Small objects: 25
# Medium objects: 21052
# Large objects: 69379
# Test
# Small objects: 0
# Medium objects: 0
# Large objects: 0
