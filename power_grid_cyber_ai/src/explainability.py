import json,sys
from pathlib import Path
import pandas as pd
from joblib import load
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR,RESULTS_DIR
def main():
 m=load(MODELS_DIR/'random_forest.joblib');f=json.loads((MODELS_DIR/'feature_order.json').read_text());o=pd.DataFrame({'feature':f,'importance':m.feature_importances_}).sort_values('importance',ascending=False).head(20);o.insert(0,'rank',range(1,len(o)+1));RESULTS_DIR.mkdir(exist_ok=True);o.to_csv(RESULTS_DIR/'top_20_features.csv',index=False);print(o.to_string(index=False))
if __name__=='__main__':main()
