# Training Datasets Guide

This guide provides information on public datasets suitable for training the Lock-In focus monitoring model.

## Recommended Datasets

### 1. State Farm Distracted Driver Detection

**Source**: [Kaggle Competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

**Description**: 
- 22,424 training images and 79,726 test images
- 10 classes of driver behavior
- High-quality images captured from vehicles

**Relevant Classes for Lock-In**:
- `c0`: Safe driving (→ `focused`)
- `c1`: Texting - right hand (→ `using_phone`)
- `c2`: Talking on phone - right hand (→ `using_phone`)
- `c3`: Texting - left hand (→ `using_phone`)
- `c4`: Talking on phone - left hand (→ `using_phone`)
- `c5`: Operating radio (→ `looking_away`)
- `c6`: Drinking (→ `looking_away`)
- `c7`: Reaching behind (→ `looking_away`)
- `c8`: Hair and makeup (→ `looking_away`)
- `c9`: Talking to passenger (→ `looking_away`)

**How to Use**:
```bash
# 1. Download from Kaggle
kaggle competitions download -c state-farm-distracted-driver-detection

# 2. Organize into Lock-In classes
python scripts/reorganize_statefarm.py --input imgs/train --output training_data
```

**License**: Competition data - review Kaggle terms

---

### 2. YawDD - Yawning Detection Dataset

**Source**: [IEEE DataPort](http://ieee-dataport.org/1096)

**Description**:
- Videos and frames of drivers yawning and not yawning
- Multiple camera angles
- Day and night lighting conditions

**Relevant Classes**:
- `Yawn` (→ `yawning`)
- `No Yawn` (can contribute to `focused`)

**How to Use**:
```bash
# Extract frames from videos
python scripts/extract_yawdd_frames.py --input YawDD/videos --output training_data/yawning
```

**License**: Research and educational use

---

### 3. Columbia Gaze Dataset

**Source**: [Columbia CAVE Lab](http://www.cs.columbia.edu/CAVE/databases/columbia_gaze/)

**Description**:
- 5,880 images from 56 people
- Different head poses and gaze directions
- Controlled lighting

**Relevant Classes**:
- Forward gaze (→ `focused`)
- Diverted gaze (→ `looking_away`)

**How to Use**:
```bash
# Process gaze annotations
python scripts/process_columbia_gaze.py --input columbia_gaze --output training_data
```

**License**: Research use only

---

### 4. ULG Multimodality Drowsiness Database

**Source**: [ULG Database](https://www.uliege.be/en/drowsiness-detection)

**Description**:
- Videos of subjects in various alertness states
- Infrared and visible light recordings
- Labeled drowsiness levels

**Relevant Classes**:
- Alert (→ `focused`)
- Drowsy (→ `sleepy`)

**License**: Academic research

---

## Dataset Preparation

### General Structure

Organize your dataset as follows:

```
training_data/
├── focused/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── looking_away/
│   ├── img_001.jpg
│   └── ...
├── using_phone/
│   └── ...
├── yawning/
│   └── ...
└── sleepy/
    └── ...
```

### Preprocessing Script

```python
# scripts/preprocess_dataset.py
import cv2
from pathlib import Path

def preprocess_image(img_path, output_path, target_size=224):
    """Standardize image size and format"""
    img = cv2.imread(str(img_path))
    img_resized = cv2.resize(img, (target_size, target_size))
    cv2.imwrite(str(output_path), img_resized)

# Process all images
input_dir = Path("raw_data")
output_dir = Path("training_data")

for class_dir in input_dir.iterdir():
    if class_dir.is_dir():
        output_class_dir = output_dir / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in class_dir.glob("*.jpg"):
            output_path = output_class_dir / img_path.name
            preprocess_image(img_path, output_path)
```

---

## Creating Your Own Dataset

### Option 1: Automated Collection

Use the provided script to collect samples:

```bash
# Collect for each class
python scripts/capture_samples.py --class_name focused --duration 120 --interval 2
python scripts/capture_samples.py --class_name looking_away --duration 60 --interval 2
python scripts/capture_samples.py --class_name using_phone --duration 60 --interval 2
python scripts/capture_samples.py --class_name yawning --duration 30 --interval 2
python scripts/capture_samples.py --class_name sleepy --duration 30 --interval 2
```

### Option 2: Manual Annotation

1. Record videos of yourself in different states
2. Extract frames using `ffmpeg`:
   ```bash
   ffmpeg -i video.mp4 -vf fps=1 frames/img_%04d.jpg
   ```
3. Manually sort frames into class folders

### Best Practices for Personal Data

1. **Lighting Variety**: Capture in different lighting (morning, afternoon, evening)
2. **Camera Angles**: Match your typical workspace setup
3. **Natural Behavior**: Act naturally, don't over-exaggerate
4. **Multiple Sessions**: Spread collection over several days
5. **Sufficient Samples**: Aim for 200+ per class minimum

---

## Dataset Augmentation

The training script applies automatic augmentation:

- **Horizontal Flip**: Increases variety
- **Rotation**: ±10 degrees
- **Color Jitter**: Brightness, contrast, saturation
- **Random Crop**: (optional)

Configure in `config.yaml`:

```yaml
training:
  augmentation:
    horizontal_flip: true
    rotation_degrees: 10
    color_jitter: true
    random_crop: false
```

---

## Combining Datasets

### Strategy 1: Mixed Training

Combine public and personal data:

```bash
# Merge datasets
cp -r public_data/* training_data/
cp -r personal_data/* training_data/
```

### Strategy 2: Transfer Learning

1. Train on large public dataset
2. Fine-tune on small personal dataset

```bash
# Initial training
python -m src.train --data_dir public_data --config config.yaml

# Fine-tuning
python -m src.train --data_dir personal_data --config config_finetune.yaml --resume models/distraction_classifier.pt
```

---

## Dataset Statistics

Recommended minimum samples per class:

| Class | Minimum | Good | Excellent |
|-------|---------|------|-----------|
| focused | 500 | 2000 | 5000+ |
| looking_away | 300 | 1500 | 3000+ |
| using_phone | 300 | 1500 | 3000+ |
| yawning | 200 | 1000 | 2000+ |
| sleepy | 200 | 1000 | 2000+ |

---

## Validation and Testing

Always split your data:

- **Training**: 70-80%
- **Validation**: 10-15%
- **Testing**: 10-15%

```python
from sklearn.model_selection import train_test_split

# Split dataset
train_val, test = train_test_split(samples, test_size=0.15, stratify=labels)
train, val = train_test_split(train_val, test_size=0.15, stratify=train_val_labels)
```

---

## Ethical Considerations

When using datasets:

1. **Review License**: Ensure you can use the data for your purpose
2. **Privacy**: Don't share personal data publicly
3. **Bias**: Be aware of demographic biases in datasets
4. **Citation**: Cite dataset sources in research
5. **Consent**: Only use images where subjects consented

---

## Dataset Quality Checklist

✅ Balanced class distribution  
✅ Variety in lighting conditions  
✅ Multiple subjects (if public data)  
✅ High image quality (not blurry)  
✅ Consistent camera angle  
✅ Natural poses (not staged)  
✅ Sufficient samples per class  
✅ Separate test set  

---

## Troubleshooting

### Issue: Imbalanced Classes

**Solution**: Use class weights or oversampling

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

### Issue: Poor Generalization

**Solution**: 
- Collect more diverse data
- Increase augmentation
- Fine-tune on personal samples

### Issue: Low Accuracy on Rare Classes

**Solution**:
- Collect more samples for underrepresented classes
- Apply targeted augmentation

---

## Resources

- [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
- [Data Augmentation Techniques](https://github.com/aleju/imgaug)
- [Labelbox](https://labelbox.com/) - Annotation tool
- [CVAT](https://github.com/opencv/cvat) - Open-source annotation

---

**Need help with datasets? Open an issue on GitHub!**

