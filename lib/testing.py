import csv
import numpy as np
from collections import defaultdict

import torch

from data import CLASSES


def eval_seg(models, dataloader, device):
    results = defaultdict(list)

    # If passed a single model rather than a list (to be ensembled) then make a list of len 1
    if type(models) not in (list, tuple):
        models = [models]

    for model in models:
        model.eval()

    for i_batch, (x, y_true, filepath, label_name) in enumerate(dataloader):
        with torch.no_grad():
            y_pred_models = []
            for model in models:
                x = x.to(device, non_blocking=True)
                y_pred = model(x)
                y_pred_models.append(y_pred)
            y_pred_models = torch.stack(y_pred_models)
            y_pred = torch.mean(y_pred_models, dim=0)

            y_pred_id = torch.argmax(y_pred, dim=1).cpu().numpy()

            y_true = y_true.numpy()

            results['y_true'].extend(y_true)
            results['y_pred_prob'].extend(torch.softmax(y_pred, dim=1).cpu().numpy())
            results['y_pred'].extend(y_pred_id)
            results['filepath'].extend(filepath)
            results['labelname'].extend(label_name)

    results['y_pred_prob'] = np.stack(results['y_pred_prob'])

    return results


def eval_unetseg(model, unet_model, dataloader, stack, device, clahe_transform=None):
    results = defaultdict(list)
    model.eval()
    unet_model.eval()

    for i_batch, (x, y_true, filepath, label_name) in enumerate(dataloader):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            mask = torch.sigmoid(unet_model(x)) > 0.5

            if stack:
                x_m = x.clone()
                x_m[~mask] = 0
                x = torch.cat((x_m, x), dim=1)
            else:
                x[~mask] = 0

            if clahe_transform is not None:
                for i_img, img in enumerate(x):
                    x[i_img][0] = torch.tensor(clahe_transform(image=(x[i_img][0].cpu().numpy() * 255).astype(np.uint8))['image']/255).float()

            y_pred = model(x)
            y_pred_id = torch.argmax(y_pred, dim=1).cpu().numpy()
        y_true = y_true.numpy()

        results['y_true'].extend(y_true)
        results['y_pred_prob'].extend(torch.softmax(y_pred, dim=1).cpu().numpy())
        results['y_pred'].extend(y_pred_id)
        results['filepath'].extend(filepath)
        results['labelname'].extend(label_name)

    return results


def eval_ensemble(classifier_model, unet_model, unet_classifier_model, dataloader, stack, device, ensemble_weight=1, clahe_transform=None):
    results = defaultdict(list)
    classifier_model.eval()
    unet_model.eval()
    unet_classifier_model.eval()

    for i_batch, (x, y_true, filepath, label_name) in enumerate(dataloader):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)

            # simple classifier method
            y_pred_classifier = classifier_model(x)

            # unet method
            if stack:
                mask = torch.sigmoid(unet_model(x)) > 0.5
                x_m = x.clone()
                x_m[~mask] = 0
                x = torch.cat((x_m, x), dim=1)
            else:
                mask = torch.sigmoid(unet_model(x)) > 0.5
                x[~mask] = 0

            if clahe_transform is not None:
                for i_img, img in enumerate(x):
                    x[i_img][0] = torch.tensor(clahe_transform(image=(x[i_img][0].cpu().numpy() * 255).astype(np.uint8))['image']/255).float()

            y_pred_unet = unet_classifier_model(x)

        y_pred = (torch.softmax(y_pred_unet, dim=1)*ensemble_weight + torch.softmax(y_pred_classifier, dim=1)) / (1+ensemble_weight)
        y_pred_id = torch.argmax(y_pred, dim=1).cpu().numpy()

        y_true = y_true.numpy()

        results['y_true'].extend(y_true)
        results['y_pred_prob'].extend(torch.softmax(y_pred, dim=1).cpu().numpy())
        results['y_pred'].extend(y_pred_id)
        results['filepath'].extend(filepath)
        results['labelname'].extend(label_name)

    return results

def write_predictions_to_csv(results_dict, csv_path):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['true_id', 'true_name', 'pred_id', 'pred_name'])
        writer.writerows(zip(results_dict['y_true'],
                             [CLASSES[v] for v in results_dict['y_true']],
                             results_dict['y_pred'],
                             [CLASSES[v] for v in results_dict['y_pred']]))