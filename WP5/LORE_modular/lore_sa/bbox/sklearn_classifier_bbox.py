from lore_sa.bbox.bbox import AbstractBBox

__all__ = ["AbstractBBox","sklearnBBox"]
class sklearnBBox(AbstractBBox):
    def __init__(self, classifier):
        self.bbox = classifier

    def predict(self, X):
        return self.bbox.predict(X)

    def predict_proba(self, X):
        return self.bbox.predict_proba(X)