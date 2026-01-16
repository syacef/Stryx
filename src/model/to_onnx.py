from dataset.supervised_dataset import SafariSupervisedDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_ds = SafariSupervisedDataset("./data/sa_fari_train_ext.json", "./data")
    val_ds = SafariSupervisedDataset("./data/sa_fari_test_ext.json", "./data")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)

    num_species = len(train_ds.species_map)

    print("Showing species mapping:")
    for species_id, species_name in train_ds.species_map.items():
        print(f"  {species_id}: {species_name}")

    # ssl_model = DistilledTinyViT()
    # ssl_model.load_state_dict(torch.load("student_epoch6_tinyvit.pth", map_location="cpu"))
    # ssl_model.eval()
    # torch_model = SafariSpeciesClassifier(ssl_model, embedding_dim=384, num_species=num_species)
    # torch_model.load_state_dict(torch.load("best_safari_model.pth", map_location="cpu"))
    # torch_model.eval()

    # # Create example inputs for exporting the model. The inputs should be a tuple of tensors.
    # example_inputs = (torch.randn(1, 8, 3, 224, 224),)  # Example input shape: (batch_size, num_frames, channels, height, width)
    # onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
    # onnx_program.save("safari_clf_epoch10.onnx")
