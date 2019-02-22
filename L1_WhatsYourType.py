from fastai.metrics import error_rate
from fastai.vision import *
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
import pokepy

client = pokepy.V2Client(cache='in_disk', cache_location='/Users/mwyang/Downloads')
img_dir = '/Users/mwyang/Downloads/pokemon'
img_path = Path(img_dir)
fnames = os.listdir(img_path)


def process_image_fname(f):
    id = f[:-4]
    if '-' not in id:
        try:
            poke = client.get_pokemon(id)
            return (poke[0].name, [t.type.name for t in poke[0].types])
        except:
            pass
    return (None, None)


df = [process_image_fname(f) for f in tqdm(fnames)]
df = pd.DataFrame(df, columns=['name', 'types'])

df['fname'] = fnames
df['id'] = [f[:-4] for f in fnames]
pd.to_numeric(df['id'], errors='coerce')
df.sort_values('id', inplace=True)
df.reset_index(inplace=True)

df['primary_type'] = df.types.map(lambda x: x[0] if x is not None else None)

data = ImageDataBunch.from_df(
    img_path,
    df[~df.isnull().any(axis=1)],
    fn_col='fname', label_col='types', ds_tfms=get_transforms(), size=224)
