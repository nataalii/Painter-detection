import torch
from torch.utils.data import DataLoader
from src.data_preproccesing import load_data, split_data
from src.model_training import CNN, train_loop, test_loop
from src.model_evaluation import evaluate_model, plot_confusion_matrix, plot_metrics
from src.predict import predict_painter
import torchmetrics
from sklearn.metrics import classification_report, confusion_matrix



def main():
    # Data preparation
    datapath = './paintings_dataset/images'
    dataset = load_data(datapath)
    train_dataset, test_dataset = split_data(dataset)
    painters = dataset.classes
    painters = [p.replace('_', ' ') for p in painters]

    # Model training
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes=len(painters))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy(task='multiclass', num_classes=len(painters)).to(device)

    epochs = 35
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        print(f'Epoch [{epoch+1}/{epochs}]')
        train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer, metric, device)
        test_loss, test_acc = test_loop(test_loader, model, metric, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    torch.save(model.state_dict(), 'your_trained_model.pth')
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

    # # Model evaluation
    model.load_state_dict(torch.load('your_trained_model.pth'))
    all_labels, all_preds = evaluate_model(model, test_loader, device)
    print(classification_report(all_labels, all_preds, target_names=painters))
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(conf_matrix, painters)


    # Example prediction
    image_path = './goya.jpg' 
    painter = predict_painter(image_path, model, painters, device)
    print(f"The predicted painter for {image_path} is: {painter}")

if __name__ == "__main__":
    main()
