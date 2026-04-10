from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class egoexo4dDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        # self._embed = hub.load("../egoexo4d_rlds_dataset_builder")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist/Ego camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='7D state.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='7D action.',
                    ),
                    'discount': tfds.features.Scalar(dtype=np.float32),
                    'reward': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'language_instruction': tfds.features.Text(),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='USE-large 512D embedding.',
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(doc='Path to the original data file.'),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            # 'train': self._generate_examples(path='/path/to/egoexo4d'),
            'val': self._generate_examples(path='/path/to/egoexo4d'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            print(episode_path)            
            data = np.load(episode_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.tolist()

            episode = []
            T = len(data)
            for i, step in enumerate(data):
                lang = step.get('language_instruction', '')
                language_embedding = self._embed([lang])[0].numpy()

                img = step['image'].astype(np.uint8)
                wrist = step.get('wrist_image')
                if wrist is None:
                    wrist = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    wrist = wrist.astype(np.uint8)

                state = step.get('state', np.zeros(7, dtype=np.float32)).astype(np.float32)
                action = step.get('action', np.zeros(7, dtype=np.float32)).astype(np.float32)

                episode.append({
                    'observation': {
                        'image': img,
                        'wrist_image': wrist,
                        'state': state,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(i == (T - 1)),
                    'is_first': i == 0,
                    'is_last': i == (T - 1),
                    'is_terminal': i == (T - 1),
                    'language_instruction': lang,
                    'language_embedding': language_embedding,
                })

            sample = {
                'steps': episode,
                'episode_metadata': {'file_path': episode_path},
            }
            return episode_path, sample

        episode_paths = glob.glob(path)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(sorted(episode_paths)) | beam.Map(_parse_example)
        for p in episode_paths:
            yield _parse_example(p)
