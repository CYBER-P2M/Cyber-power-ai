from pathlib import Path
import csv,random,shutil
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/vision/dataset_raw'; OUT=ROOT/'data/vision/substation_dataset'; CL=ROOT/'data/vision/classes.txt'; SOURCES=['agv_day','agv_night_dark','agv_night_light','misc']; EXT={'.jpg','.jpeg','.png','.bmp','.webp'}
def main():
    if not CL.exists(): raise FileNotFoundError(f"Copy classes.txt to {CL}")
    classes=[x.strip() for x in CL.read_text(encoding='utf-8').splitlines() if x.strip()]
    pairs=[]
    for src in SOURCES:
        sd=RAW/src; ld=sd/'labels'
        if not sd.exists() or not ld.exists(): continue
        for im in sd.iterdir():
            if im.is_file() and im.suffix.lower() in EXT and (ld/f'{im.stem}.txt').exists(): pairs.append((src,im,ld/f'{im.stem}.txt'))
    if not pairs: raise RuntimeError('No image-label pairs found')
    random.Random(42).shuffle(pairs); n=len(pairs); a=int(n*.8); b=a+int(n*.1); splits={'train':pairs[:a],'val':pairs[a:b],'test':pairs[b:]}
    for typ in ['images','labels']:
        if (OUT/typ).exists(): shutil.rmtree(OUT/typ)
        for sp in splits: (OUT/typ/sp).mkdir(parents=True,exist_ok=True)
    rows=[]
    for sp,items in splits.items():
        for src,im,lab in items:
            stem=f'{src}_{im.stem}'; shutil.copy2(im,OUT/'images'/sp/f'{stem}{im.suffix.lower()}'); shutil.copy2(lab,OUT/'labels'/sp/f'{stem}.txt'); rows.append([sp,src,im.name,f'{stem}{im.suffix.lower()}',f'{stem}.txt'])
    (OUT/'data.yaml').write_text('path: data/vision/substation_dataset\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {x}\n' for i,x in enumerate(classes)),encoding='utf-8')
    with open(OUT/'dataset_manifest.csv','w',newline='',encoding='utf-8') as f: csv.writer(f).writerows([['split','source','original_image','output_image','label']]+rows)
    print('pairs:',n,'train:',len(splits['train']),'val:',len(splits['val']),'test:',len(splits['test']))
if __name__=='__main__': main()
