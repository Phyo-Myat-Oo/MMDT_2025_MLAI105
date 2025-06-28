# CNN Models Comparison on tkl image dataset
[By Thein Kyaw Lwin]

## Experiment
Firstly, I collected 10 images online and labelled manually. Then, I modified the Tr's code to adapt on my own dataset. Predictions results from each model are listed below.


|    | ResNet50           | VGGNet16           | InceptionV3   | ConvNeXt           | EfficientNet     |   ResNet50_prob |   VGGNet16_prob |   InceptionV3_prob |   ConvNeXt_prob |   EfficientNet_prob | label    |
|----|--------------------|--------------------|---------------|--------------------|------------------|-----------------|-----------------|--------------------|-----------------|---------------------|----------|
|  0 | Great_Pyrenees     | Maltese_dog        | web_site      | tub                | mask             |        0.249248 |       0.0936128 |           0.999959 |        0.331172 |           0.0979043 | baby     |
|  1 | sports_car         | sports_car         | web_site      | sports_car         | grille           |        0.213542 |       0.499425  |           1        |        0.671603 |           0.166676  | car      |
|  2 | soccer_ball        | soccer_ball        | web_site      | soccer_ball        | soccer_ball      |        0.841772 |       0.870668  |           1        |        0.945497 |           0.86134   | football |
|  3 | pomegranate        | pomegranate        | flatworm      | Granny_Smith       | wooden_spoon     |        0.655223 |       0.913065  |           1        |        0.568936 |           0.0838462 | apple    |
|  4 | acoustic_guitar    | electric_guitar    | web_site      | electric_guitar    | microphone       |        0.73337  |       0.95669   |           1        |        0.894345 |           0.09645   | guitar   |
|  5 | tabby              | tabby              | flatworm      | lynx               | tabby            |        0.684788 |       0.43925   |           0.999997 |        0.387555 |           0.298807  | cat      |
|  6 | electric_fan       | electric_fan       | web_site      | electric_fan       | electric_fan     |        0.99791  |       0.998452  |           1        |        0.949404 |           0.885767  | fan      |
|  7 | Labrador_retriever | Labrador_retriever | web_site      | Labrador_retriever | golden_retriever |        0.242197 |       0.294892  |           0.999908 |        0.401643 |           0.666864  | dog      |
|  8 | alp                | alp                | flatworm      | alp                | alp              |        0.774512 |       0.803675  |           1        |        0.742693 |           0.682475  | mountain |
|  9 | stopwatch          | analog_clock       | web_site      | analog_clock       | strainer         |        0.561455 |       0.552459  |           1        |        0.928616 |           0.326287  | clock    |

## Model comparison
As per the requirements of the assignment, I compared the performance of the models using metrics such as:

1. Inference time per image
2. Prediction accuracy
3. Model size and memory usage

After manually checking, some predicted labels are acceptable even though they are not the exact matched with actual labels. So, I decided to consider that acceptable labels to calcualte prediction accuracy.

| Model        |   Correct |   Total | Accuracy   | Avg Inference Time   |   Model Size (MB) | Parameters   |
|--------------|-----------|---------|------------|----------------------|-------------------|--------------|
| ResNet50     |         8 |      10 | 80.00%     | 0.4939 sec           |             98.2  | 25,636,712   |
| VGGNet16     |         8 |      10 | 80.00%     | 0.7686 sec           |            527.83 | 138,357,544  |
| InceptionV3  |         0 |      10 | 0.00%      | 0.6526 sec           |             91.66 | 23,851,784   |
| ConvNeXt     |         8 |      10 | 80.00%     | 1.4892 sec           |            109.42 | 28,589,128   |
| EfficientNet |         5 |      10 | 50.00%     | 4.0551 sec           |            255.9  | 66,658,687   |

## Top 3 Modes: ResNet50, VGGNet16, ConvNext

Some models cannot perform well on my dataset. I think that some models may perform poorly because the images in my dataset differ significantly from the types of images they were originally trained on.

[Remark] I get assistant from ChatGPT for 'model comparison' part to speed up my project submission.
