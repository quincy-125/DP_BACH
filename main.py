import sys
import tensorflow as tf

from cli import main


if __name__ == '__main__':
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.experimental.list_physical_devices('GPU'))
    assert len(tf.config.experimental.list_physical_devices('GPU')) > 0, "No GPUs found"

    sys.exit(main())