from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]/'data/vision/substation_dataset'; EXT={'.jpg','.jpeg','.png','.bmp','.webp'}
def main():
    errors=[]
    if not (ROOT/'data.yaml').exists(): errors.append('data.yaml missing')
    for sp in ['train','val','test']:
        ims={p.stem for p in (ROOT/'images'/sp).glob('*') if p.suffix.lower() in EXT}; lbs={p.stem for p in (ROOT/'labels'/sp).glob('*.txt')}
        if ims-lbs: errors.append(f'{sp}: missing labels {len(ims-lbs)}')
        if lbs-ims: errors.append(f'{sp}: orphan labels {len(lbs-ims)}')
        print(sp,len(ims),'images',len(lbs),'labels')
    if errors: print('ERRORS:',*errors,sep='\n- '); sys.exit(1)
    print('OK: dataset structure is valid')
if __name__=='__main__': main()
