
from Model.preprocessing.datasetManager import DatasetManger
import tensorflow as tf
from Model.base.learning_rate_scheduler import Get_scheduler_callbacks, scheduler


def cross_validation(model, METRICS, loss,  epoch, batch_size, x, y, cv=5, test_size=0.1, lr=10e-4, Scheduler=False, verbose="auto", Shuffle=False):

    results = []
    callbacks = [Get_scheduler_callbacks(
        scheduler)] if Scheduler == True else []

    managerdataset = DatasetManger(
        x, y, batch_size=batch_size, n_splits=cv, test_size=test_size, Shuffle=Shuffle)

    for train_ds, test_ds in managerdataset.create_dataset():

        clonedModel = tf.keras.models.clone_model(model)

        opt = tf.keras.optimizers.Adam(learning_rate=lr,)
        metrics_instantiation = [m(name=m.__name__) for m in METRICS]
        loss_instanse = loss()
        clonedModel.compile(optimizer=opt, loss=loss_instanse,
                            metrics=metrics_instantiation)

        history = clonedModel.fit(
            train_ds, epochs=epoch, batch_size=batch_size, validation_data=test_ds, callbacks=callbacks, verbose=verbose)

        results.append({
            "train_set": train_ds,
            "test_set": test_ds,
            "model": clonedModel,
            "monitor": history,
            "epoch": epoch,
            "batch": batch_size,
            "test_size": test_size,
            "lr": lr
        })

    return results
