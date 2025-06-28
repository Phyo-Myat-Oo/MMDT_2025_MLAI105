# cnn_models.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import numpy as np
import pandas as pd
import time
import psutil
import gc

from keras.preprocessing.image import img_to_array
from keras.utils import load_img


# Model-specific preprocessing + decoding functions
from keras.applications.resnet50 import preprocess_input as preprocess_resnet, decode_predictions as decode_resnet
from keras.applications.vgg16    import preprocess_input as preprocess_vgg16,  decode_predictions as decode_vgg16
from keras.applications.inception_v3 import preprocess_input as preprocess_inception, decode_predictions as decode_inception
from keras.applications.convnext import preprocess_input as preprocess_convnext, decode_predictions as decode_convnext
from keras.applications.efficientnet import preprocess_input as preprocess_efficient, decode_predictions as decode_efficient

# Map model names to their preprocess & decode functions
PREPROCESS_FN = {
    'ResNet50':    (preprocess_resnet, decode_resnet),
    'VGGNet16':    (preprocess_vgg16, decode_vgg16),
    'InceptionV3': (preprocess_inception, decode_inception),
    'ConvNeXt':    (preprocess_convnext, decode_convnext),
    'EfficientNet':(preprocess_efficient, decode_efficient)
}


class ModelPerformanceTracker:
    def get_memory_usage_mb(self):
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)

    def calculate_model_size_and_params(self, model):
        # Total parameters using model.count_params()
        total = model.count_params()
        # Trainable parameters by summing products of shapes
        trainable = int(
            sum(np.prod(w.shape) for w in model.trainable_weights)
        )
        non_trainable = total - trainable
        # Approximate size in MB (float32 = 4 bytes)
        size_mb = total * 4 / (1024 * 1024)
        return {
            'total_params': total,
            'trainable_params': trainable,
            'non_trainable_params': non_trainable,
            'model_size_mb': size_mb
        }

    def calculate_accuracy_if_labels_known(self, preds, truths):
        if truths is None:
            return None
        correct = sum(1 for p, t in zip(preds, truths) if p.lower() in t.lower())
        return correct / len(preds) if preds else 0.0


class cnnModels:
    """Loads pretrained models and exposes methods to classify images."""
    def __init__(self):
        self.tracker = ModelPerformanceTracker()
        self.models = {}
        self.specs  = {}

        loaders = {
            'ResNet50':    keras.applications.ResNet50,
            'VGGNet16':    keras.applications.VGG16,
            'InceptionV3': keras.applications.InceptionV3,
            'ConvNeXt':    keras.applications.ConvNeXtTiny,
            'EfficientNet':keras.applications.EfficientNetB7
        }
        # Instantiate models and record specs
        for name, ctor in loaders.items():
            model = ctor(weights='imagenet')
            self.models[name] = model
            self.specs[name]  = self.tracker.calculate_model_size_and_params(model)

    def get_model(self, name):
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found.")
        return self.models[name]

    def get_model_specs(self, name):
        return self.specs.get(name, None)

    def classify_image(self, name, img):
        """Resize, preprocess, predict, and decode top-1."""
        model = self.get_model(name)
        img_resized = img.resize((model.input_shape[1], model.input_shape[2]))
        x = img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        preprocess_input, decode_fn = PREPROCESS_FN[name]
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)
        return decode_fn(preds, top=1)


def get_predictions_with_performance_metrics(image_dir, true_labels=None):
    """
    Processes a folder of images through all models, measuring:
    - classification & probability
    - inference time
    - memory usage
    - (optional) accuracy if true_labels provided
    Returns: (predictions_df, performance_df)
    """
    cm = cnnModels()
    model_names = list(cm.models.keys())

    # Gather image paths and labels
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    labels = [os.path.splitext(f)[0] for f in files]

    # Prepare results storage
    pred_data = {'label': labels}
    perf_data = {}

    for name in model_names:
        model = cm.get_model(name)
        preprocess_input, decode_fn = PREPROCESS_FN[name]

        class_preds, prob_preds, times = [], [], []
        mem_before = cm.tracker.get_memory_usage_mb()

        for fname in files:
            img = load_img(os.path.join(image_dir, fname), target_size=(model.input_shape[1], model.input_shape[2]))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            t0 = time.perf_counter()
            preds = model.predict(x, verbose=0)
            t1 = time.perf_counter()
            times.append(t1 - t0)

            dec = decode_fn(preds, top=1)[0][0]
            class_preds.append(dec[1])
            prob_preds.append(float(dec[2]))

        mem_after = cm.tracker.get_memory_usage_mb()
        specs   = cm.get_model_specs(name)
        accuracy = cm.tracker.calculate_accuracy_if_labels_known(class_preds, true_labels)

        perf_data[name] = {
            'avg_time_ms':    np.mean(times)*1000,
            'std_time_ms':    np.std(times)*1000,
            'throughput_ips': len(files)/sum(times),
            'memory_mb_before': mem_before,
            'memory_mb_after':  mem_after,
            'memory_diff_mb':   mem_after - mem_before,
            'accuracy':         accuracy,
            **specs
        }
        pred_data[name]       = class_preds
        pred_data[f"{name}_prob"] = prob_preds

    preds_df = pd.DataFrame(pred_data)
    perf_df  = pd.DataFrame(perf_data).T

    return preds_df, perf_df


def save_comprehensive_results(real_preds, fake_preds, real_perf, fake_perf, output_dir='./results'):
    """Save prediction and performance DataFrames to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    real_preds.to_csv(os.path.join(output_dir, 'real_result.csv'), index=False)
    fake_preds.to_csv(os.path.join(output_dir, 'fake_result.csv'), index=False)
    real_perf.to_csv(os.path.join(output_dir, 'real_performance_metrics.csv'))
    fake_perf.to_csv(os.path.join(output_dir, 'fake_performance_metrics.csv'))

    # Combined comparison
    comp = real_perf[['avg_time_ms','throughput_ips','total_params','model_size_mb','accuracy']].copy()
    comp.columns = ['real_'+c for c in comp.columns]
    fake_subset = fake_perf[['avg_time_ms','throughput_ips','accuracy']].copy()
    fake_subset.columns = ['fake_'+c for c in fake_subset.columns]
    comp = comp.join(fake_subset)
    comp.to_csv(os.path.join(output_dir, 'model_performance_comparison.csv'))
