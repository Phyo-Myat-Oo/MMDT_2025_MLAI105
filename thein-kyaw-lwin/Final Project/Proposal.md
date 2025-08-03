# Final Project Proposal

## DL-Based Geocoding for Myanmar Address (Address > Latitude, Longitude)

### 1. Problem Statement

The main goal of this project is to predict the geolocation information (Latitude and Longitude) from the given addresses written in Burmese (Unicode) using Deep Learning and Neural Language Processing. As of now, we have to rely on different sources to identify the unstructured Burmese address to identify the real location on the map.

- Population: Structured and Unstructured Burmese address strings used in humanitarians, logistics and public services.
- Sample: A sample of ~100,000 labeled addresses from Geospatial Data from MIMU (Myanmar Information Management Unit) + Data from OSM (Open Street Map)


### 2. Inputs, Outputs

Inputs

Address written in Burmese (Unicode):

Samples:
- ဘုရားလမ်း၊ ရန်အောင်(၁) ရပ်ကွက် ပျဥ်းမနားမြို့
- မန္တလေးမြို့၊ အောင်မြေသာစံ၊ မြောက်ပြင်၊ အိုးဘိုရပ်၊ အိုးဘိုထောင်
- မန္တလေးမြို့ ၊ ပရိယတ္တိအသင်း ၊ စာဖြေဌာန

Outputs

- Latitude
- Longitude

### 3. Dataset

Training Dataset will be created using the data from MIMU and OSM: 

MIMU Data 
- Points: (villages, wards, towns, hospitals, airport, school)
- Polygon (boundry): MIMU Administrative Boundaries (VT, TS, DT, ST)

OSM Data 
- Points: (amenities)
- Lines: Street (pending)

### 4. Expectation

- Mean Error Distance 
- Threshold Distance (eg, Within 500m)

