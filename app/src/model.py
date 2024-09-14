# IMPORT PYTORCH LIGHTNING LIBRARY APIs
import timm
import torch
import torch.nn as nn
import lightning as L

class cnnModel(L.LightningModule):
    def __init__(
        self,
        model_name="timm/tf_efficientnet_b4.ns_jft_in1k",
        learning_rate=1e-4,
        num_classes=1,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Load a pre-trained model from timm
        self.model = timm.create_model(model_name, pretrained=True)
        num_in_features = self.model.get_classifier().in_features

        ################# New Head ########################
        self.model.classifier = nn.Sequential(
            nn.Dropout(
                0.25
            ),  # Move dropout before batch norm for potential regularization
            nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=self.num_classes, bias=False),
        )  # modify classify's head

        self.save_hyperparameters()  # tell the model to save the hyperparameters into the checkpoint file

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return torch.sigmoid(self.forward(x))  # Sigmoid for probability prediction
