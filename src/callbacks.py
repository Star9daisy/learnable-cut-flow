from keras.callbacks import Callback

from .models import LearnableCutFlowModel


class ImportanceUpdate(Callback):
    def __init__(self, learnable_cut_flow: LearnableCutFlowModel):
        super().__init__()
        self.learnable_cut_flow = learnable_cut_flow

    def on_epoch_end(self, epoch, logs=None):
        # Update importance scores after each epoch
        for i, cut in enumerate(self.learnable_cut_flow.cuts):
            cut.importance_score.assign(self.learnable_cut_flow.importance.scores[i])
