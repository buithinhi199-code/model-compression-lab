def main() -> None:
    from torchvision import datasets, transforms

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    image, label = dataset[0]

    print("Number of training samples:", len(dataset))
    print("Image shape:", image.shape)
    print("Label:", label)


if __name__ == "__main__":
    main()
