from pickle import dump

from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostedDecisionTree(GradientBoostingClassifier):
    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_depth=3,
        min_impurity_decrease=0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=0.0001,
        ccp_alpha=0,
        name="gradient_boosted_decision_tree",
        input_shape=None,
    ) -> None:
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )
        self.name = name
        self.input_shape = input_shape

    def compile(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        kwargs.pop("batch_size", None)
        kwargs.pop("verbose", "auto")
        kwargs.pop("steps", None)
        kwargs.pop("callbacks", None)
        return super().predict_proba(*args, **kwargs)[:, [1]]

    def fit(self, *args, **kwargs):
        kwargs.pop("x", None)
        kwargs.pop("y", None)
        kwargs.pop("batch_size", None)
        kwargs.pop("epochs", 1)
        kwargs.pop("verbose", "auto")
        kwargs.pop("callbacks", None)
        kwargs.pop("validation_split", 0.0)
        kwargs.pop("validation_data", None)
        kwargs.pop("shuffle", True)
        kwargs.pop("class_weight", None)
        kwargs.pop("sample_weight", None)
        kwargs.pop("initial_epoch", 0)
        kwargs.pop("steps_per_epoch", None)
        kwargs.pop("validation_steps", None)
        kwargs.pop("validation_batch_size", None)
        kwargs.pop("validation_freq", 1)
        return super().fit(*args, **kwargs)

    def summary(self):
        return super().get_params()

    def save(self, path: str):
        with open(path, "wb") as f:
            dump(self, f, protocol=5)
