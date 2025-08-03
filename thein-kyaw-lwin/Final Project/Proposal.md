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
- မန္တလေးမြို့ ၊ ပရိယတ္တိအသင်း ၊ စာဖြေဌာနအနီး

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

### 5. To Discuss

- Full address combination needed ? (villages + village tract + township + district + state and region)
- If needed, combination variations (villages + village tract + township) and (villages + township) and many more
- or seperated column -> to model ?


### 6. Woking Progress

Data from OSM: myanmar-latest.osm.pbf

**Amenities per State/Region:**
| ST_PCODE   | ST           |   Amenity Count |
|------------|--------------|-----------------|
| MMR009     | Magway       |            7660 |
| MMR010     | Mandalay     |            4499 |
| MMR011     | Mon          |            4051 |
| MMR013     | Yangon       |            3615 |
| MMR005     | Sagaing      |            3293 |
| MMR006     | Tanintharyi  |            1368 |
| MMR014     | Shan (South) |             822 |
| MMR007     | Bago (East)  |             754 |
| MMR015     | Shan (North) |             711 |
| MMR017     | Ayeyarwady   |             576 |
| MMR001     | Kachin       |             553 |
| MMR012     | Rakhine      |             398 |
| MMR018     | Nay Pyi Taw  |             271 |
| MMR003     | Kayin        |             246 |
| MMR008     | Bago (West)  |             234 |
| MMR016     | Shan (East)  |             185 |
| MMR002     | Kayah        |              47 |
| MMR004     | Chin         |              33 |
|            | Total        |           29316 |

**Amenities by Type:**
| amenity_type               |   Count |
|----------------------------|---------|
| place_of_worship           |   12827 |
| monastery                  |    3502 |
| school                     |    3415 |
| hospital                   |    1439 |
| restaurant                 |    1008 |
| clinic                     |     518 |
| marketplace                |     468 |
| convenience                |     370 |
| community_centre           |     358 |
| police                     |     281 |
| pharmacy                   |     274 |
| cafe                       |     274 |
| townhall                   |     253 |
| bank                       |     246 |
| bus_station                |     225 |
| fuel                       |     209 |
| clothes                    |     184 |
| mobile_phone               |     162 |
| university                 |     155 |
| yes                        |     149 |
| library                    |     144 |
| fire_station               |     138 |
| post_office                |     133 |
| electronics                |     133 |
| fast_food                  |     111 |
| hardware                   |     107 |
| chemist                    |     107 |
| tea                        |     103 |
| supermarket                |      93 |
| kindergarten               |      90 |
| ferry_terminal             |      79 |
| wholesale                  |      70 |
| dentist                    |      67 |
| doctors                    |      64 |
| water                      |      62 |
| prison                     |      59 |
| courthouse                 |      58 |
| social_facility            |      49 |
| crematorium                |      46 |
| car_repair                 |      43 |
| copyshop                   |      41 |
| fashion                    |      39 |
| atm                        |      37 |
| college                    |      37 |
| jewelry                    |      37 |
| beauty                     |      34 |
| furniture                  |      28 |
| cinema                     |      28 |
| books                      |      27 |
| bakery                     |      27 |
| electrical                 |      27 |
| general                    |      26 |
| hairdresser                |      24 |
| department_store           |      23 |
| gift                       |      22 |
| car_parts                  |      21 |
| food_court                 |      21 |
| car                        |      21 |
| coffee                     |      20 |
| events_venue               |      19 |
| veterinary                 |      19 |
| bar                        |      19 |
| photo                      |      17 |
| computer                   |      16 |
| pub                        |      16 |
| car_wash                   |      16 |
| shoes                      |      16 |
| variety_store              |      16 |
| motorcycle                 |      15 |
| optician                   |      15 |
| mall                       |      13 |
| biergarten                 |      13 |
| beverages                  |      13 |
| school_vocational          |      13 |
| theatre                    |      12 |
| bureau_de_change           |      12 |
| stationery                 |      11 |
| seafood                    |      11 |
| car_rental                 |      11 |
| cosmetics                  |      11 |
| alcohol                    |      11 |
| garden_centre              |      11 |
| pawnbroker                 |      10 |
| trade                      |      10 |
| travel_agency              |      10 |
| bicycle                    |      10 |
| ticket                     |       9 |
| houseware                  |       9 |
| drinking_water             |       8 |
| ice_cream                  |       7 |
| animal_shelter             |       7 |
| language_school            |       7 |
| motorcycle_repair          |       7 |
| agrarian                   |       7 |
| fishing                    |       7 |
| meditation_centre          |       6 |
| research_institute         |       6 |
| food                       |       6 |
| tailor                     |       6 |
| taxi                       |       6 |
| social_centre              |       6 |
| parking                    |       6 |
| clock                      |       5 |
| florist                    |       5 |
| toilets                    |       5 |
| funeral_directors          |       5 |
| greengrocer                |       5 |
| animal_breeding            |       5 |
| shelter                    |       5 |
| bookmaker                  |       5 |
| dairy                      |       4 |
| appliance                  |       4 |
| vehicle_inspection         |       4 |
| fixme                      |       4 |
| sports                     |       4 |
| prep_school                |       4 |
| nursing_home               |       4 |
| studio                     |       4 |
| water_point                |       4 |
| medical_supply             |       4 |
| casino                     |       4 |
| radiotechnics              |       4 |
| waste_disposal             |       4 |
| gas                        |       4 |
| dry_cleaning               |       4 |
| massage                    |       4 |
| nightclub                  |       3 |
| tyres                      |       3 |
| carpet                     |       3 |
| driving_school             |       3 |
| doityourself               |       3 |
| refugee_site               |       3 |
| religion                   |       3 |
| art                        |       3 |
| childcare                  |       3 |
| grave_yard                 |       3 |
| fountain                   |       3 |
| polling_station            |       2 |
| public_bookcase            |       2 |
| school_other               |       2 |
| farm                       |       2 |
| government                 |       2 |
| boat_sharing               |       2 |
| closes                     |       2 |
| deli                       |       2 |
| fabric                     |       2 |
| bed                        |       2 |
| fashion_accessories        |       2 |
| funeral_hall               |       2 |
| grocery                    |       2 |
| kiosk                      |       2 |
| laundry                    |       2 |
| lottery                    |       2 |
| wine                       |       2 |
| toys                       |       2 |
| waste_transfer_station     |       2 |
| watches                    |       1 |
| watering_place             |       1 |
| စတိတ်ခုံငှား                      |       1 |
| telephone                  |       1 |
| vending_machine            |       1 |
| spices                     |       1 |
| trishaws stand             |       1 |
| train_station              |       1 |
| temple                     |       1 |
| accessories                |       1 |
| internet_cafe              |       1 |
| rice                       |       1 |
| marina                     |       1 |
| interior_decoration        |       1 |
| hostel                     |       1 |
| hearing_aids               |       1 |
| electronics_repair         |       1 |
| curtain                    |       1 |
| car_pooling                |       1 |
| cabinet_maker              |       1 |
| butcher                    |       1 |
| bus_stop                   |       1 |
| boutique                   |       1 |
| bicycle_repair_station     |       1 |
| bathroom_furnishing        |       1 |
| bag                        |       1 |
| baby_goods                 |       1 |
| atm;bank                   |       1 |
| astrologer                 |       1 |
| arts_centre                |       1 |
| lighting                   |       1 |
| market                     |       1 |
| ranger_station             |       1 |
| meditation_center          |       1 |
| pottery                    |       1 |
| place_of_worship;monastery |       1 |
| pet                        |       1 |
| paint                      |       1 |
| outpost                    |       1 |
| outdoor                    |       1 |
| nutrition_supplements      |       1 |
| newsagent                  |       1 |
| musical_instrument         |       1 |
| music_school               |       1 |
| motorcycle_rental          |       1 |
| mortuary                   |       1 |
| monument                   |       1 |
| money_transfer             |       1 |
| mobile_money_agent         |       1 |
| military_surplus           |       1 |
| military                   |       1 |
| သင်္ကန်းဆိုင်                     |       1 |
| Total                      |   29316 |

<img width="4500" height="4500" alt="image" src="https://github.com/user-attachments/assets/f8d026b7-21d3-43b6-8ab3-1b5dd635f8c2" />

