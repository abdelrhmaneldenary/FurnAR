import matplotlib.pyplot as plt

def train_colorization_dpt():
    # 🔥 UPDATE THIS PATH to your dataset of colorful images 🔥
    IMAGE_DIR = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}\n")

    print("Loading dataset...")
    dataset = ColorizationDataset(IMAGE_DIR, img_size=(224, 224))
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, shuffle=True, batch_size=8) # Can usually push batch size higher for colorization
    test_loader = DataLoader(test_data, shuffle=False, batch_size=8)

    print("\nInitializing DPT Model...")
    encoder = PretrainedViTEncoder()
    
    # Start with Frozen ViT
    for param in encoder.parameters():
        param.requires_grad = False
        
    model = DPT(vit_encoder=encoder).to(device)

    # Hyperparameters
    n_epochs = 5
    unfreeze_epoch = 3
    base_lr = 2e-4 
    
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=base_lr)
    # L1 Loss (Mean Absolute Error) is actually better than MSE for colorization 
    # to avoid muddy, desaturated colors.
    criterion = nn.L1Loss() 

    best_test_loss = float('inf')
    history_train_loss = []
    history_test_loss = []

    print("\nStarting Training Pipeline...")
    for epoch in range(n_epochs):
        
        if epoch == unfreeze_epoch:
            print("\n🔓 UNFREEZING THE ViT BACKBONE FOR FINE-TUNING!")
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
            print("📉 Learning rate dropped to 1e-5.\n")
        
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        for batch in loop:
            # x is the 3-channel grayscale, y is the 2-channel ab color target
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Test] "):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        history_test_loss.append(avg_test_loss)
        
        print(f"--> Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            print(f"🌟 New best model! Saving state_dict...")
            torch.save(model.state_dict(), "best_dpt_colorizer.pth")
        print("-" * 50 + "\n")

    print("Generating Training History Artifact...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), history_train_loss, label='Train Loss (MAE)', color='blue', marker='o')
    plt.plot(range(1, n_epochs + 1), history_test_loss, label='Test Loss (MAE)', color='orange', marker='s')
    plt.axvline(x=unfreeze_epoch + 1, color='red', linestyle='--', label='ViT Unfrozen')
    plt.title('DPT Colorization Training History')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('dpt_colorization_training.png', dpi=300, bbox_inches='tight')
    plt.show()

train_colorization_dpt()