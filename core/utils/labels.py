from dataclasses import dataclass


@dataclass
class Label:
    name: str
    id: int
    full_id: int
    amodal_id: int


LABELS = [
    #     name                    id  full_id, amodal_id
    Label('unlabeled',             0,       0,         0),
    Label('ego vehicle',           1,       0,         0),
    Label('rectification border',  2,       0,         0),
    Label('out of roi',            3,       0,         0),
    Label('static',                4,       0,         0),
    Label('dynamic',               5,       0,         0),
    Label('ground',                6,       0,         0),
    Label('road',                  7,       1,         0),
    Label('sidewalk',              8,       2,         0),
    Label('parking',               9,       0,         0),
    Label('rail track',           10,       0,         0),
    Label('building',             11,       3,         0),
    Label('wall',                 12,       4,         0),
    Label('fence',                13,       5,         0),
    Label('guard rail',           14,       0,         0),
    Label('bridge',               15,       0,         0),
    Label('tunnel',               16,       0,         0),
    Label('polegroup',            17,       6,         0),
    Label('pole',                 18,       6,         0),
    Label('traffic light',        19,       7,         0),
    Label('traffic sign',         20,       8,         0),
    Label('vegetation',           21,       9,         0),
    Label('terrain',              22,      10,         0),
    Label('sky',                  23,      11,         0),
    Label('person',               24,      12,         1),
    Label('rider',                25,      13,         2),
    Label('car',                  26,      14,         3),
    Label('truck',                27,      15,         4),
    Label('bus',                  28,      16,         5),
    Label('caravan',              29,      17,         6),
    Label('trailer',              30,      18,         7),
    Label('train',                31,      19,         8),
    Label('motor',                32,      20,         9),
    Label('bike',                 33,      21,        10),
    Label('road line',            34,       0,         0),
    Label('other',                35,       0,         0),
    Label('water',                36,       0,         0),
]

ID_MAP_FULL = {label.id: label.full_id for label in LABELS}
ID_MAP_AMODAL = {label.id: label.amodal_id for label in LABELS}

N_CLASSES_FULL = len(set(x.full_id for x in LABELS))
N_CLASSES_AMODAL = len(set(x.amodal_id for x in LABELS))
