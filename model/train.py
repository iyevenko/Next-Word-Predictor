import argparse
import datetime
import importlib
import os

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


@tf.function
def preprocess_dataset(ds, vocab, batch_size, window_size):
    # Extract all sentences from dataset
    ds = ds.map(lambda s: s.get('text'))
    ds = ds.map(lambda s: tf.strings.regex_replace(s, '_START_ARTICLE_\\n(.+)\\n', ''))
    ds = ds.map(lambda s: tf.strings.regex_replace(s, '_START_SECTION_\\n(.+)\\n', ''))
    ds = ds.map(lambda s: tf.strings.regex_replace(s, '_START_PARAGRAPH_', ''))
    ds = ds.map(lambda s: tf.strings.regex_replace(s, '\\n', ''))
    ds = ds.map(lambda s: tf.strings.regex_replace(s, '_NEWLINE_', ' '))
    ds = ds.map(lambda s: tf.strings.split(s, '.'))
    ds = ds.map(lambda s: tf.squeeze(s, 0))
    ds = ds.unbatch()
    ds = ds.filter(lambda s: not tf.equal(tf.strings.length(s), 0))

    # Window the data for getting next word
    @tf.function
    def make_windows(s):
        ds = tf.data.Dataset.from_tensor_slices(s)
        ds = ds.filter(lambda s: not tf.equal(tf.strings.length(s), 0))
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda x: x)
        ds = ds.batch(window_size)

        # Universal sentence encoder doesn't encode precise position of words very well
        # so I add feed the last word into the model as well as the whole sentence in order
        # to avoid common grammar mistakes like repeating words or stuff like "the of"
        prev_words = ds.map(lambda s: (tf.strings.reduce_join(s[:-1], separator=' '), s[-2]))
        next_label = ds.map(lambda s: vocab[s[-1]])
        next_label = next_label.map(lambda l: vocab[tf.constant('<UNK>')] if l == vocab.size()-1 else l)

        # Originally had this to remove <UNK> prediction but the model was way too
        # biased towards common words like 'the' 'of' 'and' etc.
        # next_label = next_label.filter(lambda l: l < vocab.size()-1)

        # Return a dataset of ((sentence, last_word)), next_word_label) tuples
        ds = zip(prev_words, next_label)
        return ds

    ds = ds.map(lambda s: tf.strings.split(s, ' '))
    ds = ds.flat_map(make_windows)
    ds = ds.shuffle(100000)
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds


def dataset_fn(tfds_path, split, vocab, batch_size, window_size):
    dir_name = os.path.join(tfds_path, 'wiki40b', 'en')

    download = True
    if os.path.exists(dir_name) and os.path.isdir(dir_name):
        download = False

    ds_splits, ds_info = tfds.load('wiki40b/en',
                    split=split,
                    batch_size=1,
                    download=download,
                    data_dir=tfds_path,
                    with_info=True)

    dataset = {}
    split_iter = iter(ds_info.splits.keys())

    for ds in ds_splits:
        split_name = next(split_iter)
        processed_ds = preprocess_dataset(ds, vocab, batch_size, window_size)
        dataset[split_name] = processed_ds

    return dataset


def load_encoder(vocab_size, tfhub_cache_dir='~/tensorflow_hub_cache', vocab_dir=None):
    # Load the model from tfhub into specified directory
    os.environ['TFHUB_CACHE_DIR'] = tfhub_cache_dir
    encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # Freeze pretrained weights since the model is way too big try to fine tune
    encoder = hub.KerasLayer(encoder, trainable=False)

    tf_hub_path = '/Users/iyevenko/tensorflow_hub_cache'
    MODEL_PATH = [d.path for d in os.scandir(tf_hub_path) if d.is_dir()][0]

    # Traverse graph to find vocab tensor used by Universal Sentence Encoder
    loader_impl = importlib.import_module('tensorflow.python.saved_model.loader_impl')
    saved_model = loader_impl.parse_saved_model(MODEL_PATH)

    graph = saved_model.meta_graphs[0].graph_def
    function = graph.library.function

    vocab_tensor = function[5].node_def[1].attr.get('value').tensor
    vocab_list = [i.decode('utf-8') for i in vocab_tensor.string_val[:vocab_size]]

    if vocab_dir is not None:
        with open(os.path.join(vocab_dir, 'vocab-{0}.txt'.format(vocab_size)), 'w') as f:
            for w in vocab_list:
                f.write(w+'\n')

    vocab_tensor = tf.constant(vocab_list)

    table_init = tf.lookup.KeyValueTensorInitializer(
        keys=vocab_tensor,
        values=tf.range(vocab_size, dtype=tf.int64),
        value_dtype=tf.int64
    )
    vocab_table = tf.lookup.StaticVocabularyTable(table_init, 1)

    return encoder, vocab_table, vocab_tensor


class NextWordPredictor(tf.keras.Model):

    def __init__(self, encoder, vocab_table, vocab_tensor, num_layers=2, d_model=512, dropout_rate=0.5):
        super().__init__()

        self.encoder = encoder
        self.vocab_size = vocab_table.size()-1
        self.vocab_tensor = vocab_tensor

        self.num_layers = num_layers

        self.dropout_layers = [
            tf.keras.layers.Dropout(dropout_rate)
            for _ in range(num_layers)
        ]

        # Lots of probelems with overfitting on this model, so added regularization to
        # prevent the model from outputting most common words every time
        self.dense_layers = [
            tf.keras.layers.Dense(d_model, activation='relu')
            for _ in range(num_layers)
        ]
        self.dense_out = tf.keras.layers.Dense(self.vocab_size)


    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs: Previous sentence -> (batch_size, d_model)
        :param training: Whether or not to apply dropout -> Bool
        :param mask: Optional mask to apply to inputs
        :return: Logits for next word -> (batch_size, vocab_size)
        '''
        sentence, last_word = inputs

        sentence_embedding = self.encoder(tf.squeeze(sentence, 1))
        last_word_embedding = self.encoder(tf.squeeze(last_word, 1))

        # Adding this skyrocketed the model's performance
        X = tf.keras.layers.concatenate([sentence_embedding, last_word_embedding])

        for i in range(self.num_layers):
            X = self.dense_layers[i](X)
            X = self.dropout_layers[i](X, training=training)

        logits = self.dense_out(X)

        return logits


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds-path', type=str, dest='ds_path')
    parser.add_argument('--model-dir', type=str, dest='model_dir')
    parser.add_argument('--log-dir', type=str, dest='log_dir')
    parser.add_argument('--epochs', type=int, default=10, dest='epochs')
    parser.add_argument('--batch-size', type=int, default=64, dest='batch_size')
    parser.add_argument('--window-size', type=int, default=5, dest='window_size')

    args = parser.parse_args()

    tfds_path = args.ds_path

    # Full training set contains about 11M (prev_words, next_word) pairs after windowing
    split = ['test',
             'train',
             'validation']

    encoder, vocab_table, vocab_list = load_encoder(10000,  vocab_dir=os.path.join(args.model_dir, 'vocab'))
    model = NextWordPredictor(encoder, vocab_table, vocab_list, d_model=512, num_layers=4)
    dataset = dataset_fn(tfds_path, split, vocab_table, args.batch_size, args.window_size)

    test_ds = dataset['test']
    train_ds = dataset['train']
    val_ds = dataset['validation']

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.metrics.SparseCategoricalAccuracy(),
                      tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
                  ])

    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=1000, profile_batch=0, histogram_freq=1)

    model.fit(train_ds, epochs=args.epochs, steps_per_epoch=20000,
              validation_data=val_ds, validation_steps=2000,
              callbacks=[tensorboard_callback])

    export_path = os.path.join(args.model_dir, 'next_word_predictor')
    tf.keras.models.save_model(model, export_path)

    result = model.evaluate(test_ds, steps=2000)

    print('Test Loss: ' + str(result[0]))
    print('Test Accuracy: ' + str(result[1]))

