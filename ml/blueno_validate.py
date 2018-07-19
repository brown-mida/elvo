import os
import logging
import keras

from blueno.elasticsearch import search_top_models
from blueno import types, preprocessing, utils
from models.luke import resnet
from generators.luke import standard_generators
from blueno_eval import evaluate_model


def get_models_to_train(data_dir):
    docs = search_top_models('http://104.196.51.205',
                             lower='0.85', upper='0.93')
    docs_to_train = []
    for doc in docs:
        data = {}
        data['purported_accuracy'] = doc.best_val_acc
        data['purported_loss'] = doc.best_val_loss
        data['date'] = doc.created_at
        data['job_name'] = doc.job_name

        # Dataset information
        if 'gcs_url' in doc:
            gcs_url = doc.gcs_url
        else:
            folder = doc.data_dir.split('/')[-2]
            # data_dir naming structure is super inconsistent.
            # Here are some edge cases to cover weird names.
            if folder == 'processed':
                folder = 'processed-sumera-1'

            # If not preprocessed (using numpy_compressed), skip
            if folder == 'numpy':
                continue

            gcs_url = 'gs://elvos/processed/{}'.format(folder)

        folder = os.path.join(data_dir, gcs_url.split('/')[-1])
        data['local_dir'] = folder
        data_info = {
            'data_dir': os.path.join(folder,
                                     'arrays/'),
            'labels_path': os.path.join(folder,
                                        'labels.csv'),
            'index_col': 'Anon ID',
            'label_col': 'occlusion_exists',
            'gcs_url': gcs_url
        }

        # Model information
        model = {
            'dropout_rate1': doc.dropout_rate1,
            'dropout_rate2': doc.dropout_rate2,
            'optimizer': keras.optimizers.Adam(lr=1e-5),
            'loss': keras.losses.categorical_crossentropy,
            'freeze': False,
            'model_callable': resnet,
        }

        # Generator / Image transformation information
        generator = {}
        generator_params = ['rotation_range', 'width_shift_range',
                            'height_shift_range', 'shear_range',
                            'zoom_range', 'horizontal_flip',
                            'vertical_flip']
        default_params = [30, 0.1, 0.1, 0, 0.1, True, False]
        for param, default_value in zip(generator_params, default_params):
            if param in doc:
                generator[param] = doc[param]
            else:
                generator[param] = default_value

        generator['generator_callable'] = standard_generators

        params = {
            'data': types.DataConfig(**data_info),
            'model': types.ModelConfig(**model),
            'generator': types.GeneratorConfig(**generator),
            'batch_size': doc.batch_size,
            'seed': 999,
            'val_split': 0.1,
            'max_epochs': 100,
            'early_stopping': True,
            'reduce_lr': False,
            'job_fn': None,
            'job_name': None
        }

        data['params'] = types.ParamConfig(**params)
        docs_to_train.append(data)
    return docs_to_train


def __get_data_if_not_exists(gcs_dir, local_dir):
    if not os.path.isdir(local_dir):
        logging.info('Dataset does not exist. Downloading from GCS...')
        os.mkdir(local_dir)
        exit = os.system('gsutil rsync -r -d {} {}'.format(gcs_dir, local_dir))
        return exit == 0
    return True


def __load_data(params):
    return preprocessing.prepare_data(params)


def __train_model(params, x_train, y_train, x_valid, y_valid):
    train_gen, valid_gen = params.generator.generator_callable(
        x_train, y_train,
        x_valid, y_valid,
        params.batch_size,
        **params.generator.__dict__
    )

    model = params.model.model_callable(input_shape=x_train.shape[1:],
                                        num_classes=y_train.shape[1],
                                        **params.model.__dict__)
    metrics = ['acc',
               utils.sensitivity,
               utils.specificity,
               utils.true_positives,
               utils.false_negatives]
    model.compile(optimizer=params.model.optimizer,
                  loss=params.model.loss,
                  metrics=metrics)
    callbacks = utils.create_callbacks(x_train, y_train, x_valid, y_valid,
                                       early_stopping=params.early_stopping,
                                       reduce_lr=params.reduce_lr)
    history = model.fit_generator(train_gen,
                                  epochs=params.max_epochs,
                                  validation_data=valid_gen,
                                  verbose=2,
                                  callbacks=callbacks)
    return model, history


if __name__ == '__main__':
    models = get_models_to_train('../tmp/')
    for model in models:
        params = model['params']
        __get_data_if_not_exists(params.data.gcs_url,
                                 model['local_dir'])
        (x_train, x_valid, x_test, y_train, y_valid, y_test,
         _, _, _) = __load_data(params)
        model, history = __train_model(params, x_train, y_train,
                                       x_valid, y_valid)
        result = evaluate_model(x_test, y_test, model,
                                normalize=True, x_train=x_train)
        print(result)
        throw
