import pytorch_lightning as pl
import torch
import wandb


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, coder = None, max_num_samples=32):
        super().__init__()
        self.max_num_samples = max_num_samples
        self.val_images = val_samples[0]
        self.val_labels = val_samples[1]
        self.coder = coder

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger:
            # Bring the tensors to CPU
            val_images = self.val_images.to(device=pl_module.device)
            val_labels = self.val_labels.to(device=pl_module.device)

            # Get model prediction
            logits = pl_module(val_images)
            preds = torch.argmax(logits, -1)
            # Log the images as wandb Image
            trainer.logger.experiment.log({
                "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                             for x, pred, y in zip(val_images[:self.max_num_samples],
                                                   preds[:self.max_num_samples],
                                                   val_labels[:self.max_num_samples])
                            ],
                "conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=val_labels.numpy(), preds=preds.numpy(),
                        class_names=list(self.coder.classes_)
                    ), 
                "Val confusion Matrix" : wandb.sklearn.plot_confusion_matrix(
                        y_true = val_labels.numpy(), y_pred = preds.numpy(), labels = list(self.coder.classes_)
                    ), 
                "roc" : wandb.plot.roc_curve(
                        y_true=val_labels.numpy(), y_probas = logits.numpy(), labels = list(self.coder.classes_)
                    )
            })