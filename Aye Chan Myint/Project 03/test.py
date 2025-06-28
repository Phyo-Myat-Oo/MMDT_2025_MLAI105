import cnn_models
import pandas as pd
from keras.utils import load_img #type: ignore
import os,time
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import plot_model
wd = os.getcwd()
print(wd)
# wd=  os.path.join(wd,'MMDT_2025_MLAI105','Aye Chan Myint','Project 03')
fake_dir = os.path.join(wd,'dataset','synthetic')
image_dir = fake_dir

model = cnn_models.cnnModels()
model_name = ['ResNet50', 'VGGNet16', 'InceptionV3', 'ConvNeXt', 'EfficientNet']
# result_df = pd.DataFrame(columns = model_name + [name + '_prob' for name in model_name])
# Create column names for top-3 predictions and probabilities
class_cols = [f"{m}_top{i}" for m in model_name for i in range(1, 4)]
prob_cols = [f"{m}_prob{i}" for m in model_name for i in range(1, 4)]
addoncolnames = [name + "_training time" for name in model_name]
result_df = pd.DataFrame(columns=class_cols + prob_cols + ['label'])
    
labels = []    
row_values = []
addon_values = []

for filename in os.listdir(image_dir):
    if filename.endswith('.jpeg') or filename.endswith('.png')or filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        img = load_img(image_path)   
        labels.append(filename.split('.')[0])
        truelabel = filename.split(".")[0]

        image_class_preds = []
        image_prob_preds = []
        pred_times = []
        accuracylist = []
        for name in model_name:
            start_time = time.time()
            # preds = model.classify_image(name, img)[0][0][1:3]
            preds = model.classify_image(name, img)[0]
            inference_time = time.time() - start_time
            pred_times.append(inference_time)
            for i in range(3):
                print(f'{i}: {preds[i][1]}\t{preds[i][2]}')
                image_class_preds.append(preds[i][1])
                image_prob_preds.append(preds[i][2])
            # class_preds.append(preds[0])
            # prob_preds.append(preds[1])
        # row_values.append(class_preds + prob_preds)
        row_values.append(image_class_preds + image_prob_preds+ [filename.split('.')[0]])
        addon_values.append(pred_times)
# for name in model_name:   
#     m =model.get_model(name)     
#     plt.figure()
#     plot_model(m, to_file=os.path.join(wd,'results',f'{name}.png'), show_shapes=True, show_layer_names=True)
# result_df1 = pd.DataFrame(row_values, columns = model_name + [name + '_prob' for name in model_name]) 
# result_df1['label'] = labels     
# addon_df = pd.DataFrame(addon_values, columns = addoncolnames)
# resultdf = pd.concat([result_df1,addon_df],axis=1)

result_df = pd.DataFrame(row_values, columns=class_cols + prob_cols + ['label'])        
addondf = pd.DataFrame(addon_values, columns=addoncolnames)
resultdf = pd.concat([result_df, addondf], axis=1)

# Plot the training time with larger fonts
plt.figure(figsize=(14, 8))

# Set global font scaling (optional)
plt.rcParams.update({'font.size': 12})  # Base size for all text

time_cols = [name + "_training time" for name in model_name]
for model in time_cols:
    sns.lineplot(x=resultdf['label'], y=resultdf[model], 
                label=model.replace('_training time', ''),
                linewidth=2)  # Thicker lines for better visibility

# Customize individual elements with larger fonts
plt.title('Inference Times per Image', fontsize=16, pad=20)  # pad adds title spacing
plt.ylabel('Seconds', fontsize=14)
plt.xlabel('Image Label', fontsize=14)
plt.xticks(rotation=90, fontsize=13)  # Rotated x-labels
plt.yticks(fontsize=12)  # Y-axis ticks
plt.legend(fontsize=12, framealpha=1)  # Make legend more readable

# Adjust layout and save
plt.tight_layout()
plt.savefig(os.path.join(wd, 'results', 'synthetictrainingtime1.png'), 
           dpi=300, bbox_inches='tight')  # Higher DPI for quality
# plt.close()  # Prevent display if running in script

cnnmodels = cnn_models.cnnModels()
print(cnnmodels.get_model_stats())
stats =cnnmodels.get_model_stats()

print("Model Stats:")
modelsize = []
memoryusage = []
modename=[]
for name, values in stats.items():
    print(f"{name}: Size = {values['ModelSize_MB']} MB, RAM = {values['MemoryUsage_MB']} MB")
    modelsize.append(values['ModelSize_MB'])
    memoryusage.append(values['MemoryUsage_MB'])
    modename.append(name)
modelstats = pd.DataFrame()
modelstats['Model']=modename
modelstats['Model Size (MB)']=modelsize
modelstats['Memory Usage (MB)']=memoryusage
plt.figure(figsize=(10, 6))
sns.barplot(data=modelstats, x='Model', y='Model Size (MB)', palette='Blues_d')
plt.title('Model Size Comparison')
plt.ylabel('Size (MB)')
plt.xlabel('CNN Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(wd,'results','modelsize.png'))
plt.figure(figsize=(10, 6))
sns.barplot(data=modelstats, x='Model', y='Memory Usage (MB)', palette='Blues_d')
plt.title('Memory Usage Comparison')
plt.ylabel('Size (MB)')
plt.xlabel('CNN Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(wd,'results','memory usage.png'))
print('a')