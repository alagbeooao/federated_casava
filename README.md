# Federated learning for cassava disease diagnosis

Ten simulated farms train one cassava leaf-disease classifier without pooling their images. The notebook then checks how far the result can be trusted: what federation costs in accuracy, how the worst-served farm does, whether the model's confidence is calibrated, whether it is less certain when it is wrong, and whether a model small enough for a phone holds up.

## Data

iCassava 2019: 5,656 labelled leaf photographs in five classes (bacterial blight, brown streak disease, green mottle, mosaic disease and healthy), plus about 16,000 unlabelled images. The data is not included here.

The farms are simulated by splitting the labelled set with a Dirichlet prior on each farm's disease mix (α = 0.4 in the main run). This tests the training methods, not a real deployment.

## What the notebook does

- ResNet-18 pretrained on ImageNet, with BatchNorm statistics frozen so nothing farm-specific is averaged
- Federated averaging with server momentum (FedAvgM) and plain SGD on each farm, 30 rounds
- A centralised baseline trained on the pooled data, for comparison
- Temperature scaling fitted across farms without pooling their validation data
- MC-dropout uncertainty and a risk-coverage curve
- Grad-CAM to check what the model looks at
- Distillation into MobileNetV3-Small on unlabelled images, then INT8 quantisation
- A heterogeneity sweep over α = 10, 1, 0.4 and 0.1
- Client-level differential privacy with an RDP accountant (switched off in this run)

## Results (one seed)

| Model | Test accuracy | Macro F1 |
|---|---|---|
| Always predict the most common class | 47.0% | – |
| Centralised | 82.2% | 0.759 |
| Federated, α = 0.4 | 78.4% | 0.662 |
| Distilled MobileNetV3-Small (6.2 MB) | 77.0% | 0.633 |
| Distilled + INT8 (4.4 MB, 9 ms per image on CPU) | 77.2% | – |

Heterogeneity sweep, macro F1 at the final round: 0.685 at α = 10, 0.660 at α = 1, 0.662 at α = 0.4 and 0.296 at α = 0.1. Results move by about 0.05 from round to round, so averages over the last five rounds are a fairer comparison: 0.667, 0.635, 0.613 and 0.306. Federated training stays reasonably close to centralised until farms become highly specialised, then collapses.

Other results:

- Worst farm (test set reweighted to that farm's disease mix): 61.7% federated, 67.4% centralised
- Calibration error (ECE) fell from 0.054 to 0.026 after temperature scaling
- Answering only the most certain 49% of cases cut the error rate from 22% to 7%
- In the Grad-CAM examples, attention falls on lesions and leaf tissue rather than the background

## Running it

Open the notebook in Colab on a GPU runtime (an L4 is enough), put the archive at `MyDrive/Project/cassava-disease.zip` and choose Run all. The main run takes about 30 minutes and the sweep about an hour more. Every federated run saves a checkpoint after each round, so after a disconnect Run all carries on where it stopped. `cfg.smoke_test = True` runs the whole pipeline on 1,200 images in about five minutes.

## Limitations

The farms are slices of one dataset, so connectivity, farms joining and leaving, and differences in cameras between farms are not modelled. Only the disease mix varies between farms. All results come from a single seed and partition. The images are RGB only.
