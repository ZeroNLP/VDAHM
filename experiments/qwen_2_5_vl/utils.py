# utils.py
def get_layers(model):
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    else:
        return model.model.layers